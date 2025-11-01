from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import time
import threading
import json
import re
from datetime import datetime
from trading_engine import TradingEngine
from market_data import MarketDataFetcher
from ai_trader import AITrader
from database import Database
from version import __version__, __github_owner__, __repo__, GITHUB_REPO_URL, LATEST_RELEASE_URL
from exchange_service import ExchangeCredentials, create_exchange_client, ExchangeClientError

app = Flask(__name__)
CORS(app)

db = Database('AITradeGame.db')
market_fetcher = MarketDataFetcher()
trading_engines = {}
auto_trading = True
TRADE_FEE_RATE = 0.001  # 默认交易费率
DEFAULT_MAX_LEVERAGE = 20


def _build_trading_engine(model: dict) -> TradingEngine:
    """Instantiate a trading engine for a given model configuration."""
    provider = db.get_provider(model['provider_id'])
    if not provider:
        raise ValueError('Provider not found for model')

    ai_trader = AITrader(
        api_key=model['api_key'],
        api_url=model['api_url'],
        model_name=model['model_name']
    )

    trading_mode = (model.get('trading_mode') or 'manual').lower()
    quote_currency = (model.get('exchange_quote_currency') or 'USDT').upper()
    fee_rate = TRADE_FEE_RATE
    exchange_client = None

    if trading_mode == 'exchange':
        credentials = ExchangeCredentials(
            name=model.get('exchange_name') or '',
            api_key=model.get('exchange_api_key') or '',
            api_secret=model.get('exchange_api_secret') or '',
            passphrase=model.get('exchange_passphrase'),
            use_sandbox=bool(model.get('exchange_use_sandbox')),
            account_type=model.get('exchange_type'),
            quote_currency=quote_currency
        )

        exchange_client = create_exchange_client(credentials)

        # Refresh fee rate if not cached
        try:
            if model.get('exchange_fee_rate') is not None:
                fee_rate = float(model['exchange_fee_rate'])
            else:
                fee_rate = exchange_client.get_taker_fee(f"BTC/{quote_currency}")
                db.update_model_exchange_info(model['id'], fee_rate=fee_rate)
        except ExchangeClientError as exc:
            print(f"[WARN] Failed to fetch fee rate for model {model['id']}: {exc}")

        # Persist latest balance snapshot
        try:
            snapshot = exchange_client.get_balance_snapshot()
            total_balance = snapshot.get('total_value') or snapshot.get('total') or (
                (snapshot.get('free') or 0) + (snapshot.get('used') or 0)
            )
            db.update_model_exchange_info(model['id'], last_balance=total_balance)
        except ExchangeClientError as exc:
            print(f"[WARN] Failed to fetch balance for model {model['id']}: {exc}")

    return TradingEngine(
        model_id=model['id'],
        db=db,
        market_fetcher=market_fetcher,
        ai_trader=ai_trader,
        trade_fee_rate=fee_rate,
        trading_mode=trading_mode,
        exchange_client=exchange_client,
        quote_currency=quote_currency,
        default_max_leverage=DEFAULT_MAX_LEVERAGE
    )

@app.route('/')
def index():
    return render_template('index.html')

# ============ Provider API Endpoints ============

@app.route('/api/providers', methods=['GET'])
def get_providers():
    """Get all API providers"""
    providers = db.get_all_providers()
    return jsonify(providers)

@app.route('/api/providers', methods=['POST'])
def add_provider():
    """Add new API provider"""
    data = request.json
    try:
        provider_id = db.add_provider(
            name=data['name'],
            api_url=data['api_url'],
            api_key=data['api_key'],
            models=data.get('models', '')
        )
        return jsonify({'id': provider_id, 'message': 'Provider added successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/providers/<int:provider_id>', methods=['DELETE'])
def delete_provider(provider_id):
    """Delete API provider"""
    try:
        db.delete_provider(provider_id)
        return jsonify({'message': 'Provider deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/providers/models', methods=['POST'])
def fetch_provider_models():
    """Fetch available models from provider's API"""
    data = request.json
    api_url = data.get('api_url')
    api_key = data.get('api_key')

    if not api_url or not api_key:
        return jsonify({'error': 'API URL and key are required'}), 400

    try:
        # This is a placeholder - implement actual API call based on provider
        # For now, return empty list or common models
        models = []

        # Try to detect provider type and call appropriate API
        if 'openai.com' in api_url.lower():
            # OpenAI API call
            import requests
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            response = requests.get(f'{api_url}/models', headers=headers, timeout=10)
            if response.status_code == 200:
                result = response.json()
                models = [m['id'] for m in result.get('data', []) if 'gpt' in m['id'].lower()]
        elif 'deepseek' in api_url.lower():
            # DeepSeek API
            import requests
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            response = requests.get(f'{api_url}/models', headers=headers, timeout=10)
            if response.status_code == 200:
                result = response.json()
                models = [m['id'] for m in result.get('data', [])]
        else:
            # Default: return common model names
            models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']

        return jsonify({'models': models})
    except Exception as e:
        print(f"[ERROR] Fetch models failed: {e}")
        return jsonify({'error': f'Failed to fetch models: {str(e)}'}), 500

# ============ Model API Endpoints ============

@app.route('/api/models', methods=['GET'])
def get_models():
    models = db.get_all_models()
    return jsonify(models)

@app.route('/api/models', methods=['POST'])
def add_model():
    data = request.json
    try:
        provider = db.get_provider(data['provider_id'])
        if not provider:
            return jsonify({'error': 'Provider not found'}), 404

        trading_mode = (data.get('trading_mode') or 'manual').lower()
        exchange_name = data.get('exchange_name')
        exchange_type = data.get('exchange_type')
        exchange_api_key = data.get('exchange_api_key')
        exchange_api_secret = data.get('exchange_api_secret')
        exchange_passphrase = data.get('exchange_passphrase')
        exchange_use_sandbox = bool(data.get('exchange_use_sandbox', False))
        exchange_quote_currency = (data.get('exchange_quote_currency') or 'USDT').upper()

        initial_capital = float(data.get('initial_capital', 100000))
        exchange_last_balance = None
        exchange_fee_rate = None

        if trading_mode == 'exchange':
            if not exchange_name:
                return jsonify({'error': 'Exchange name is required for exchange mode'}), 400
            if not exchange_api_key or not exchange_api_secret:
                return jsonify({'error': 'Exchange API key and secret are required'}), 400

            credentials = ExchangeCredentials(
                name=exchange_name,
                api_key=exchange_api_key,
                api_secret=exchange_api_secret,
                passphrase=exchange_passphrase,
                use_sandbox=exchange_use_sandbox,
                account_type=exchange_type,
                quote_currency=exchange_quote_currency
            )
            try:
                exchange_client = create_exchange_client(credentials)
            except ExchangeClientError as exc:
                return jsonify({'error': f'Exchange connection failed: {exc}'}), 400

            try:
                balance_snapshot = exchange_client.get_balance_snapshot()
                exchange_last_balance = (
                    balance_snapshot.get('total_value')
                    or balance_snapshot.get('total')
                    or (
                        (balance_snapshot.get('free') or 0)
                        + (balance_snapshot.get('used') or 0)
                    )
                )
                if exchange_last_balance is not None:
                    initial_capital = float(exchange_last_balance)
            except ExchangeClientError as exc:
                exchange_client.close()
                return jsonify({'error': f'Failed to fetch exchange balance: {exc}'}), 400

            try:
                exchange_fee_rate = exchange_client.get_taker_fee(f"BTC/{exchange_quote_currency}")
            except ExchangeClientError as exc:
                print(f"[WARN] Unable to fetch exchange fee rate: {exc}")
            finally:
                exchange_client.close()
        else:
            # Ensure manual mode uses default settings
            exchange_name = None
            exchange_type = None
            exchange_api_key = None
            exchange_api_secret = None
            exchange_passphrase = None
            exchange_use_sandbox = False
            exchange_quote_currency = 'USDT'

        model_id = db.add_model(
            name=data['name'],
            provider_id=data['provider_id'],
            model_name=data['model_name'],
            initial_capital=initial_capital,
            trading_mode=trading_mode,
            exchange_name=exchange_name,
            exchange_type=exchange_type,
            exchange_api_key=exchange_api_key,
            exchange_api_secret=exchange_api_secret,
            exchange_passphrase=exchange_passphrase,
            exchange_use_sandbox=exchange_use_sandbox,
            exchange_quote_currency=exchange_quote_currency,
            exchange_last_balance=exchange_last_balance,
            exchange_fee_rate=exchange_fee_rate
        )

        model = db.get_model(model_id)
        try:
            trading_engines[model_id] = _build_trading_engine(model)
        except Exception as exc:
            print(f"[ERROR] Failed to initialize engine for model {model_id}: {exc}")
            return jsonify({'error': f'Engine initialization failed: {exc}'}), 500
        print(f"[INFO] Model {model_id} ({data['name']}) initialized")

        return jsonify({'id': model_id, 'message': 'Model added successfully'})

    except Exception as e:
        print(f"[ERROR] Failed to add model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<int:model_id>', methods=['DELETE'])
def delete_model(model_id):
    try:
        model = db.get_model(model_id)
        model_name = model['name'] if model else f"ID-{model_id}"
        
        db.delete_model(model_id)
        if model_id in trading_engines:
            engine = trading_engines.pop(model_id)
            try:
                if engine.exchange_client:
                    engine.exchange_client.close()
            except Exception as exc:
                print(f"[WARN] Failed to close exchange client for model {model_id}: {exc}")
        
        print(f"[INFO] Model {model_id} ({model_name}) deleted")
        return jsonify({'message': 'Model deleted successfully'})
    except Exception as e:
        print(f"[ERROR] Delete model {model_id} failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<int:model_id>/portfolio', methods=['GET'])
def get_portfolio(model_id):
    prices_data = market_fetcher.get_current_prices(['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE'])
    current_prices = {coin: prices_data[coin]['price'] for coin in prices_data}
    
    portfolio = db.get_portfolio(model_id, current_prices)
    model = db.get_model(model_id)

    if model and (model.get('trading_mode') or 'manual').lower() == 'exchange':
        engine = trading_engines.get(model_id)
        if engine and engine.exchange_client:
            try:
                snapshot = engine.exchange_client.get_balance_snapshot()
                available_cash = snapshot.get('available', snapshot.get('free', 0.0)) or 0.0
                portfolio['exchange_balance'] = snapshot
                portfolio['exchange_available_cash'] = available_cash
                portfolio['cash'] = float(available_cash)
                total_value = snapshot.get('total_value')
                if total_value is not None:
                    portfolio['exchange_total_value'] = float(total_value)
                    portfolio['total_value'] = float(total_value)
            except ExchangeClientError as exc:
                portfolio['exchange_balance_error'] = str(exc)
        else:
            # Fallback: fetch snapshot using a temporary client
            credentials = ExchangeCredentials(
                name=model.get('exchange_name') or '',
                api_key=model.get('exchange_api_key') or '',
                api_secret=model.get('exchange_api_secret') or '',
                passphrase=model.get('exchange_passphrase'),
                use_sandbox=bool(model.get('exchange_use_sandbox')),
                account_type=model.get('exchange_type'),
                quote_currency=(model.get('exchange_quote_currency') or 'USDT').upper()
            )
            client = None
            try:
                client = create_exchange_client(credentials)
                snapshot = client.get_balance_snapshot()
                available_cash = snapshot.get('available', snapshot.get('free', 0.0)) or 0.0
                portfolio['exchange_balance'] = snapshot
                portfolio['exchange_available_cash'] = available_cash
                portfolio['cash'] = float(available_cash)
                total_value = snapshot.get('total_value')
                if total_value is not None:
                    portfolio['exchange_total_value'] = float(total_value)
                    portfolio['total_value'] = float(total_value)
            except ExchangeClientError as exc:
                portfolio['exchange_balance_error'] = str(exc)
            finally:
                if client:
                    try:
                        client.close()
                    except Exception:
                        pass

    account_value = db.get_account_value_history(model_id, limit=100)
    
    return jsonify({
        'portfolio': portfolio,
        'account_value_history': account_value
    })

@app.route('/api/models/<int:model_id>/trades', methods=['GET'])
def get_trades(model_id):
    limit = request.args.get('limit', 50, type=int)
    trades = db.get_trades(model_id, limit=limit)
    return jsonify(trades)

@app.route('/api/models/<int:model_id>/conversations', methods=['GET'])
def get_conversations(model_id):
    limit = request.args.get('limit', 20, type=int)
    conversations = db.get_conversations(model_id, limit=limit)
    return jsonify(conversations)

@app.route('/api/aggregated/portfolio', methods=['GET'])
def get_aggregated_portfolio():
    """Get aggregated portfolio data across all models"""
    prices_data = market_fetcher.get_current_prices(['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE'])
    current_prices = {coin: prices_data[coin]['price'] for coin in prices_data}

    # Get aggregated data
    models = db.get_all_models()
    total_portfolio = {
        'total_value': 0,
        'cash': 0,
        'positions_value': 0,
        'realized_pnl': 0,
        'unrealized_pnl': 0,
        'initial_capital': 0,
        'positions': []
    }

    all_positions = {}

    for model in models:
        portfolio = db.get_portfolio(model['id'], current_prices)
        if portfolio:
            total_portfolio['total_value'] += portfolio.get('total_value', 0)
            total_portfolio['cash'] += portfolio.get('cash', 0)
            total_portfolio['positions_value'] += portfolio.get('positions_value', 0)
            total_portfolio['realized_pnl'] += portfolio.get('realized_pnl', 0)
            total_portfolio['unrealized_pnl'] += portfolio.get('unrealized_pnl', 0)
            total_portfolio['initial_capital'] += portfolio.get('initial_capital', 0)

            # Aggregate positions by coin and side
            for pos in portfolio.get('positions', []):
                key = f"{pos['coin']}_{pos['side']}"
                if key not in all_positions:
                    all_positions[key] = {
                        'coin': pos['coin'],
                        'side': pos['side'],
                        'quantity': 0,
                        'avg_price': 0,
                        'total_cost': 0,
                        'leverage': pos['leverage'],
                        'current_price': pos['current_price'],
                        'pnl': 0
                    }

                # Weighted average calculation
                current_pos = all_positions[key]
                current_cost = current_pos['quantity'] * current_pos['avg_price']
                new_cost = pos['quantity'] * pos['avg_price']
                total_quantity = current_pos['quantity'] + pos['quantity']

                if total_quantity > 0:
                    current_pos['avg_price'] = (current_cost + new_cost) / total_quantity
                    current_pos['quantity'] = total_quantity
                    current_pos['total_cost'] = current_cost + new_cost
                    current_pos['pnl'] = (pos['current_price'] - current_pos['avg_price']) * total_quantity

    total_portfolio['positions'] = list(all_positions.values())

    # Get multi-model chart data
    chart_data = db.get_multi_model_chart_data(limit=100)

    return jsonify({
        'portfolio': total_portfolio,
        'chart_data': chart_data,
        'model_count': len(models)
    })

@app.route('/api/models/chart-data', methods=['GET'])
def get_models_chart_data():
    """Get chart data for all models"""
    limit = request.args.get('limit', 100, type=int)
    chart_data = db.get_multi_model_chart_data(limit=limit)
    return jsonify(chart_data)

@app.route('/api/market/prices', methods=['GET'])
def get_market_prices():
    coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']
    prices = market_fetcher.get_current_prices(coins)
    return jsonify(prices)

@app.route('/api/models/<int:model_id>/execute', methods=['POST'])
def execute_trading(model_id):
    if model_id not in trading_engines:
        model = db.get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        try:
            trading_engines[model_id] = _build_trading_engine(model)
        except Exception as exc:
            print(f"[ERROR] Failed to initialize engine for model {model_id}: {exc}")
            return jsonify({'error': f'Engine initialization failed: {exc}'}), 500
    
    try:
        result = trading_engines[model_id].execute_trading_cycle()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def trading_loop():
    print("[INFO] Trading loop started")
    
    while auto_trading:
        try:
            if not trading_engines:
                time.sleep(30)
                continue
            
            print(f"\n{'='*60}")
            print(f"[CYCLE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"[INFO] Active models: {len(trading_engines)}")
            print(f"{'='*60}")
            
            for model_id, engine in list(trading_engines.items()):
                try:
                    print(f"\n[EXEC] Model {model_id}")
                    result = engine.execute_trading_cycle()
                    
                    if result.get('success'):
                        print(f"[OK] Model {model_id} completed")
                        if result.get('executions'):
                            for exec_result in result['executions']:
                                signal = exec_result.get('signal', 'unknown')
                                coin = exec_result.get('coin', 'unknown')
                                msg = exec_result.get('message', '')
                                if signal != 'hold':
                                    print(f"  [TRADE] {coin}: {msg}")
                    else:
                        error = result.get('error', 'Unknown error')
                        print(f"[WARN] Model {model_id} failed: {error}")
                        
                except Exception as e:
                    print(f"[ERROR] Model {model_id} exception: {e}")
                    import traceback
                    print(traceback.format_exc())
                    continue
            
            print(f"\n{'='*60}")
            print(f"[SLEEP] Waiting 3 minutes for next cycle")
            print(f"{'='*60}\n")
            
            time.sleep(180)
            
        except Exception as e:
            print(f"\n[CRITICAL] Trading loop error: {e}")
            import traceback
            print(traceback.format_exc())
            print("[RETRY] Retrying in 60 seconds\n")
            time.sleep(60)
    
    print("[INFO] Trading loop stopped")

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    models = db.get_all_models()
    leaderboard = []
    
    prices_data = market_fetcher.get_current_prices(['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE'])
    current_prices = {coin: prices_data[coin]['price'] for coin in prices_data}
    
    for model in models:
        portfolio = db.get_portfolio(model['id'], current_prices)
        account_value = portfolio.get('total_value', model['initial_capital'])
        returns = ((account_value - model['initial_capital']) / model['initial_capital']) * 100
        
        leaderboard.append({
            'model_id': model['id'],
            'model_name': model['name'],
            'account_value': account_value,
            'returns': returns,
            'initial_capital': model['initial_capital']
        })
    
    leaderboard.sort(key=lambda x: x['returns'], reverse=True)
    return jsonify(leaderboard)

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get system settings"""
    try:
        settings = db.get_settings()
        return jsonify(settings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['PUT'])
def update_settings():
    """Update system settings"""
    try:
        data = request.json
        trading_frequency_minutes = int(data.get('trading_frequency_minutes', 60))
        trading_fee_rate = float(data.get('trading_fee_rate', 0.001))

        success = db.update_settings(trading_frequency_minutes, trading_fee_rate)

        if success:
            return jsonify({'success': True, 'message': 'Settings updated successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to update settings'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/version', methods=['GET'])
def get_version():
    """Get current version information"""
    return jsonify({
        'current_version': __version__,
        'github_repo': GITHUB_REPO_URL,
        'latest_release_url': LATEST_RELEASE_URL
    })

@app.route('/api/check-update', methods=['GET'])
def check_update():
    """Check for GitHub updates"""
    try:
        import requests

        # Get latest release from GitHub
        headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'AITradeGame/1.0'
        }

        # Try to get latest release
        try:
            response = requests.get(
                f"https://api.github.com/repos/{__github_owner__}/{__repo__}/releases/latest",
                headers=headers,
                timeout=5
            )

            if response.status_code == 200:
                release_data = response.json()
                latest_version = release_data.get('tag_name', '').lstrip('v')
                release_url = release_data.get('html_url', '')
                release_notes = release_data.get('body', '')

                # Compare versions
                is_update_available = compare_versions(latest_version, __version__) > 0

                return jsonify({
                    'update_available': is_update_available,
                    'current_version': __version__,
                    'latest_version': latest_version,
                    'release_url': release_url,
                    'release_notes': release_notes,
                    'repo_url': GITHUB_REPO_URL
                })
            else:
                # If API fails, still return current version info
                return jsonify({
                    'update_available': False,
                    'current_version': __version__,
                    'error': 'Could not check for updates'
                })
        except Exception as e:
            print(f"[WARN] GitHub API error: {e}")
            return jsonify({
                'update_available': False,
                'current_version': __version__,
                'error': 'Network error checking updates'
            })

    except Exception as e:
        print(f"[ERROR] Check update failed: {e}")
        return jsonify({
            'update_available': False,
            'current_version': __version__,
            'error': str(e)
        }), 500

def compare_versions(version1, version2):
    """Compare two version strings.

    Returns:
        1 if version1 > version2
        0 if version1 == version2
        -1 if version1 < version2
    """
    def normalize(v):
        # Extract numeric parts from version string
        parts = re.findall(r'\d+', v)
        # Pad with zeros to make them comparable
        return [int(p) for p in parts]

    v1_parts = normalize(version1)
    v2_parts = normalize(version2)

    # Pad shorter version with zeros
    max_len = max(len(v1_parts), len(v2_parts))
    v1_parts.extend([0] * (max_len - len(v1_parts)))
    v2_parts.extend([0] * (max_len - len(v2_parts)))

    # Compare
    if v1_parts > v2_parts:
        return 1
    elif v1_parts < v2_parts:
        return -1
    else:
        return 0

def init_trading_engines():
    try:
        models = db.get_all_models()

        if not models:
            print("[WARN] No trading models found")
            return

        print(f"\n[INIT] Initializing trading engines...")
        for model in models:
            model_id = model['id']
            model_name = model['name']

            try:
                detailed_model = db.get_model(model_id)
                if not detailed_model:
                    print(f"  [WARN] Model {model_id} ({model_name}): Not found in database")
                    continue

                trading_engines[model_id] = _build_trading_engine(detailed_model)
                print(f"  [OK] Model {model_id} ({model_name})")
            except Exception as e:
                print(f"  [ERROR] Model {model_id} ({model_name}): {e}")
                continue

        print(f"[INFO] Initialized {len(trading_engines)} engine(s)\n")

    except Exception as e:
        print(f"[ERROR] Init engines failed: {e}\n")

if __name__ == '__main__':
    import webbrowser
    import os
    
    print("\n" + "=" * 60)
    print("AITradeGame - Starting...")
    print("=" * 60)
    print("[INFO] Initializing database...")
    
    db.init_db()
    
    print("[INFO] Database initialized")
    print("[INFO] Initializing trading engines...")
    
    init_trading_engines()
    
    if auto_trading:
        trading_thread = threading.Thread(target=trading_loop, daemon=True)
        trading_thread.start()
        print("[INFO] Auto-trading enabled")
    
    print("\n" + "=" * 60)
    print("AITradeGame is running!")
    print("Server: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # 自动打开浏览器
    def open_browser():
        time.sleep(1.5)  # 等待服务器启动
        url = "http://localhost:5000"
        try:
            webbrowser.open(url)
            print(f"[INFO] Browser opened: {url}")
        except Exception as e:
            print(f"[WARN] Could not open browser: {e}")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
