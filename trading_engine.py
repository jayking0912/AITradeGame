from datetime import datetime
from typing import Dict, Optional
import json

from exchange_service import ExchangeClient, ExchangeClientError

class TradingEngine:
    def __init__(
        self,
        model_id: int,
        db,
        market_fetcher,
        ai_trader,
        trade_fee_rate: float = 0.001,
        trading_mode: str = 'manual',
        exchange_client: Optional[ExchangeClient] = None,
        quote_currency: str = 'USDT',
        default_max_leverage: int = 20
    ):
        self.model_id = model_id
        self.db = db
        self.market_fetcher = market_fetcher
        self.ai_trader = ai_trader
        self.coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']
        self.trade_fee_rate = trade_fee_rate  # 默认费率
        self.trading_mode = trading_mode
        self.exchange_client = exchange_client
        self.quote_currency = quote_currency
        self.default_max_leverage = default_max_leverage
    
    def execute_trading_cycle(self) -> Dict:
        try:
            market_state = self._get_market_state()
            
            current_prices = {coin: market_state[coin]['price'] for coin in market_state}
            
            portfolio = self.db.get_portfolio(self.model_id, current_prices)
            portfolio = self._attach_exchange_state(portfolio)
            
            account_info = self._build_account_info(portfolio)
            
            decisions = self.ai_trader.make_decision(
                market_state, portfolio, account_info
            )
            
            self.db.add_conversation(
                self.model_id,
                user_prompt=self._format_prompt(market_state, portfolio, account_info),
                ai_response=json.dumps(decisions, ensure_ascii=False),
                cot_trace=''
            )
            
            execution_results = self._execute_decisions(decisions, market_state, portfolio)
            
            updated_portfolio = self.db.get_portfolio(self.model_id, current_prices)
            self.db.record_account_value(
                self.model_id,
                updated_portfolio['total_value'],
                updated_portfolio['cash'],
                updated_portfolio['positions_value']
            )
            
            return {
                'success': True,
                'decisions': decisions,
                'executions': execution_results,
                'portfolio': updated_portfolio
            }
            
        except Exception as e:
            print(f"[ERROR] Trading cycle failed (Model {self.model_id}): {e}")
            import traceback
            print(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_market_state(self) -> Dict:
        market_state = {}
        prices = self.market_fetcher.get_current_prices(self.coins)
        
        for coin in self.coins:
            if coin in prices:
                market_state[coin] = prices[coin].copy()
                indicators = self.market_fetcher.calculate_technical_indicators(coin)
                market_state[coin]['indicators'] = indicators
        
        return market_state

    def _attach_exchange_state(self, portfolio: Dict) -> Dict:
        if not self.exchange_client:
            portfolio['exchange_balance'] = None
            return portfolio

        try:
            snapshot = self.exchange_client.get_balance_snapshot()
            portfolio['exchange_balance'] = snapshot
            available_cash = snapshot.get('available', snapshot.get('free', 0.0)) or 0.0

            portfolio['exchange_available_cash'] = available_cash
            portfolio['cash'] = float(available_cash)

            total_value = snapshot.get('total_value')
            if total_value is not None:
                portfolio['exchange_total_value'] = float(total_value)
                portfolio['total_value'] = float(total_value)
        except ExchangeClientError as exc:
            message = f"[WARN] Exchange balance fetch failed for model {self.model_id}: {exc}"
            print(message)
            portfolio['exchange_balance'] = None
            portfolio['exchange_balance_error'] = str(exc)

        return portfolio

    def _symbol_for_coin(self, coin: str) -> str:
        return f"{coin}/{self.quote_currency.upper()}"

    def _get_fee_rate(self, coin: str) -> float:
        if not self.exchange_client:
            return self.trade_fee_rate
        try:
            symbol = self._symbol_for_coin(coin)
            return self.exchange_client.get_taker_fee(symbol)
        except ExchangeClientError as exc:
            print(f"[WARN] Fee lookup failed for {coin}: {exc}")
            return self.trade_fee_rate

    def _get_leverage_limit(self, coin: str) -> int:
        limit = None
        if self.exchange_client:
            try:
                symbol = self._symbol_for_coin(coin)
                limit = self.exchange_client.get_max_leverage(symbol)
            except ExchangeClientError as exc:
                print(f"[WARN] Leverage lookup failed for {coin}: {exc}")

        if not limit:
            limit = self.default_max_leverage

        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = self.default_max_leverage

        return max(1, limit_value)

    def _get_available_cash(self, portfolio: Dict) -> float:
        if self.exchange_client:
            available = portfolio.get('exchange_available_cash')
            if available is not None:
                try:
                    return max(0.0, float(available))
                except (TypeError, ValueError):
                    pass
        try:
            return max(0.0, float(portfolio.get('cash', 0)))
        except (TypeError, ValueError):
            return 0.0
    
    def _build_account_info(self, portfolio: Dict) -> Dict:
        model = self.db.get_model(self.model_id)
        initial_capital = model['initial_capital']
        total_value = portfolio['total_value']
        total_return = ((total_value - initial_capital) / initial_capital) * 100
        
        return {
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_return': total_return,
            'initial_capital': initial_capital,
            'exchange_balance': portfolio.get('exchange_balance')
        }
    
    def _format_prompt(self, market_state: Dict, portfolio: Dict, 
                      account_info: Dict) -> str:
        return f"Market State: {len(market_state)} coins, Portfolio: {len(portfolio['positions'])} positions"
    
    def _execute_decisions(self, decisions: Dict, market_state: Dict, 
                          portfolio: Dict) -> list:
        results = []
        
        for coin, decision in decisions.items():
            if coin not in self.coins:
                continue
            
            signal = decision.get('signal', '').lower()
            
            try:
                if signal == 'buy_to_enter':
                    result = self._execute_buy(coin, decision, market_state, portfolio)
                elif signal == 'sell_to_enter':
                    result = self._execute_sell(coin, decision, market_state, portfolio)
                elif signal == 'close_position':
                    result = self._execute_close(coin, decision, market_state, portfolio)
                elif signal == 'hold':
                    result = {'coin': coin, 'signal': 'hold', 'message': 'Hold position'}
                else:
                    result = {'coin': coin, 'error': f'Unknown signal: {signal}'}
                
                results.append(result)
                
            except Exception as e:
                results.append({'coin': coin, 'error': str(e)})
        
        return results
    
    def _execute_buy(self, coin: str, decision: Dict, market_state: Dict, 
                    portfolio: Dict) -> Dict:
        quantity = float(decision.get('quantity', 0))
        requested_leverage = int(decision.get('leverage', 1) or 1)
        leverage_limit = self._get_leverage_limit(coin)
        leverage = max(1, min(requested_leverage, leverage_limit))
        price = market_state[coin]['price']
        
        if quantity <= 0:
            return {'coin': coin, 'error': 'Invalid quantity'}
        
        # 计算交易额和交易费（按交易额的比例）
        trade_amount = quantity * price  # 交易额
        fee_rate = self._get_fee_rate(coin)
        trade_fee = trade_amount * fee_rate  # 交易费
        required_margin = (quantity * price) / leverage  # 保证金
        
        # 总需资金 = 保证金 + 交易费
        total_required = required_margin + trade_fee
        available_cash = self._get_available_cash(portfolio)
        if total_required > available_cash:
            return {
                'coin': coin,
                'error': f'Insufficient cash (including fees). Required ${total_required:.2f}, available ${available_cash:.2f}'
            }
        
        # 更新持仓
        self.db.update_position(
            self.model_id, coin, quantity, price, leverage, 'long'
        )
        
        # 记录交易（包含交易费）
        self.db.add_trade(
            self.model_id, coin, 'buy_to_enter', quantity, 
            price, leverage, 'long', pnl=0, fee=trade_fee
        )
        
        messages = [
            f'Long {quantity:.4f} {coin} @ ${price:.2f}',
            f'Fee: ${trade_fee:.2f}'
        ]
        if leverage != requested_leverage:
            messages.append(f'Leverage adjusted to {leverage}x (limit {leverage_limit}x)')
        
        return {
            'coin': coin,
            'signal': 'buy_to_enter',
            'quantity': quantity,
            'price': price,
            'leverage': leverage,
            'fee': trade_fee,  # 返回费用信息
            'fee_rate': fee_rate,
            'message': '; '.join(messages)
        }
    
    def _execute_sell(self, coin: str, decision: Dict, market_state: Dict, 
                 portfolio: Dict) -> Dict:
        quantity = float(decision.get('quantity', 0))
        requested_leverage = int(decision.get('leverage', 1) or 1)
        leverage_limit = self._get_leverage_limit(coin)
        leverage = max(1, min(requested_leverage, leverage_limit))
        price = market_state[coin]['price']
        
        if quantity <= 0:
            return {'coin': coin, 'error': 'Invalid quantity'}
        
        # 计算交易额和交易费
        trade_amount = quantity * price
        fee_rate = self._get_fee_rate(coin)
        trade_fee = trade_amount * fee_rate
        required_margin = (quantity * price) / leverage
        
        # 总需资金 = 保证金 + 交易费
        total_required = required_margin + trade_fee
        available_cash = self._get_available_cash(portfolio)
        if total_required > available_cash:
            return {
                'coin': coin,
                'error': f'Insufficient cash (including fees). Required ${total_required:.2f}, available ${available_cash:.2f}'
            }
        
        # 更新持仓
        self.db.update_position(
            self.model_id, coin, quantity, price, leverage, 'short'
        )
        
        # 记录交易（包含交易费）
        self.db.add_trade(
            self.model_id, coin, 'sell_to_enter', quantity, 
            price, leverage, 'short', pnl=0, fee=trade_fee
        )
        
        messages = [
            f'Short {quantity:.4f} {coin} @ ${price:.2f}',
            f'Fee: ${trade_fee:.2f}'
        ]
        if leverage != requested_leverage:
            messages.append(f'Leverage adjusted to {leverage}x (limit {leverage_limit}x)')

        return {
            'coin': coin,
            'signal': 'sell_to_enter',
            'quantity': quantity,
            'price': price,
            'leverage': leverage,
            'fee': trade_fee,
            'fee_rate': fee_rate,
            'message': '; '.join(messages)
        }
    
    def _execute_close(self, coin: str, decision: Dict, market_state: Dict, 
                    portfolio: Dict) -> Dict:
        position = None
        for pos in portfolio['positions']:
            if pos['coin'] == coin:
                position = pos
                break
        
        if not position:
            return {'coin': coin, 'error': 'Position not found'}
        
        current_price = market_state[coin]['price']
        entry_price = position['avg_price']
        quantity = position['quantity']
        side = position['side']
        
        # 计算平仓利润（未扣费）
        if side == 'long':
            gross_pnl = (current_price - entry_price) * quantity
        else:  # short
            gross_pnl = (entry_price - current_price) * quantity
        
        # 计算平仓交易费（按平仓时的交易额）
        trade_amount = quantity * current_price
        fee_rate = self._get_fee_rate(coin)
        trade_fee = trade_amount * fee_rate
        net_pnl = gross_pnl - trade_fee  # 净利润 = 毛利润 - 交易费
        
        # 关闭持仓
        self.db.close_position(self.model_id, coin, side)
        
        # 记录平仓交易（包含费用和净利润）
        self.db.add_trade(
            self.model_id, coin, 'close_position', quantity,
            current_price, position['leverage'], side, pnl=net_pnl, fee=trade_fee  # 新增fee参数
        )
        
        return {
            'coin': coin,
            'signal': 'close_position',
            'quantity': quantity,
            'price': current_price,
            'pnl': net_pnl,
            'fee': trade_fee,
            'fee_rate': fee_rate,
            'message': f'Close {coin}, Gross P&L: ${gross_pnl:.2f}, Fee: ${trade_fee:.2f}, Net P&L: ${net_pnl:.2f}'
        }
