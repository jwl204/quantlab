from microstructure.orderbook import LimitOrderBook, Order


def make_book():
    book = LimitOrderBook()
    book.add_limit(Order(1, "buy", 99.0, 5.0))
    book.add_limit(Order(2, "buy", 98.0, 5.0))
    book.add_limit(Order(3, "sell", 101.0, 5.0))
    book.add_limit(Order(4, "sell", 102.0, 5.0))
    return book


def test_top_of_book():
    book = make_book()
    assert book.best_bid == 99.0
    assert book.best_ask == 101.0
    assert book.mid == 100.0
    assert book.spread == 2.0


def test_market_buy_walks_price_levels():
    book = make_book()
    fills = book.market_order("buy", 7.0)  # 5 @101 then 2 @102
    assert [(f.price, f.qty) for f in fills] == [(101.0, 5.0), (102.0, 2.0)]
    assert book.best_ask == 102.0  # front level exhausted


def test_price_time_priority_is_fifo():
    book = LimitOrderBook()
    book.add_limit(Order(10, "buy", 100.0, 3.0))  # earlier
    book.add_limit(Order(11, "buy", 100.0, 3.0))  # later, same price
    fills = book.market_order("sell", 4.0)
    assert fills[0].maker_id == 10 and fills[0].qty == 3.0
    assert fills[1].maker_id == 11 and fills[1].qty == 1.0


def test_partial_fill_leaves_remainder():
    book = make_book()
    book.market_order("buy", 2.0)  # eats 2 of the 5 @101
    remaining = book.asks[101.0][0].qty
    assert remaining == 3.0


def test_cancel_removes_liquidity():
    book = make_book()
    assert book.cancel(3) is True
    assert book.best_ask == 102.0
    assert book.cancel(999) is False


def test_oversized_order_partially_fills_then_stops():
    book = make_book()
    fills = book.market_order("buy", 100.0)  # only 10 available
    assert sum(f.qty for f in fills) == 10.0
    assert book.best_ask is None
