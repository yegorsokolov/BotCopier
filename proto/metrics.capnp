@0xc2c2c2c2c2c2c2c2;

struct Metrics {
  time @0 :Text;
  magic @1 :Int32;
  winRate @2 :Float64;
  avgProfit @3 :Float64;
  tradeCount @4 :Int32;
  drawdown @5 :Float64;
  sharpe @6 :Float64;
  fileWriteErrors @7 :Int32;
  socketErrors @8 :Int32;
  bookRefreshSeconds @9 :Int32;
}
