@0xc1c1c1c1c1c1c1c1;

struct TradeEvent {
  eventId @0 :Int32;
  eventTime @1 :Text;
  brokerTime @2 :Text;
  localTime @3 :Text;
  action @4 :Text;
  ticket @5 :Int32;
  magic @6 :Int32;
  source @7 :Text;
  symbol @8 :Text;
  orderType @9 :Int32;
  lots @10 :Float64;
  price @11 :Float64;
  sl @12 :Float64;
  tp @13 :Float64;
  profit @14 :Float64;
  profitAfterTrade @15 :Float64;
  spread @16 :Float64;
  comment @17 :Text;
  remainingLots @18 :Float64;
  slippage @19 :Float64;
  volume @20 :Int32;
  openTime @21 :Text;
  bookBidVol @22 :Float64;
  bookAskVol @23 :Float64;
  bookImbalance @24 :Float64;
  slHitDist @25 :Float64;
  tpHitDist @26 :Float64;
  decisionId @27 :Int32;
  traceId @28 :Text;
  spanId @29 :Text;
}
