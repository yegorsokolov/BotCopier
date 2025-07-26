#property strict
#include "model_interface.mqh"

extern string SymbolToTrade = "EURUSD";
extern double Lots = 0.1;
extern int MagicNumber = 1234;
extern bool EnableDebugLogging = false;

double ModelCoefficients[] = {__COEFFICIENTS__};
double ModelIntercept = __INTERCEPT__;
double ModelThreshold = __THRESHOLD__;
double ProbabilityLookup[] = {__PROBABILITY_TABLE__};

int OnInit()
{
   return(INIT_SUCCEEDED);
}

//----------------------------------------------------------------------
// Feature extraction utilities
//----------------------------------------------------------------------

double GetSLDistance()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
      if(OrderSelect(i, SELECT_BY_POS) &&
         OrderMagicNumber() == MagicNumber &&
         OrderSymbol() == SymbolToTrade)
      {
         if(OrderType() == OP_BUY)
            return(Bid - OrderStopLoss());
         else
            return(OrderStopLoss() - Ask);
      }
   return(0.0);
}

double GetTPDistance()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
      if(OrderSelect(i, SELECT_BY_POS) &&
         OrderMagicNumber() == MagicNumber &&
         OrderSymbol() == SymbolToTrade)
      {
         if(OrderType() == OP_BUY)
            return(OrderTakeProfit() - Bid);
         else
            return(Ask - OrderTakeProfit());
      }
   return(0.0);
}

double GetFeature(int index)
{
   /* Return a simple set of features for the logistic model.
      Feature mapping is intentionally minimal and should be kept in
      sync with the Python training script.  Unknown indices default
      to zero. */
   switch(index)
   {
__FEATURE_CASES__      default:
         return(0.0);
   }
}

double ComputeLogisticScore()
{
   double z = ModelIntercept;
   int n = ArraySize(ModelCoefficients);
   for(int i=0; i<n; i++)
      z += ModelCoefficients[i] * GetFeature(i);
   // logistic function
   return(1.0 / (1.0 + MathExp(-z)));
}

double GetProbability()
{
   if(ArraySize(ProbabilityLookup) == 24)
      return(ProbabilityLookup[TimeHour(TimeCurrent())]);
   return(ComputeLogisticScore());
}

bool HasOpenOrders()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
      if(OrderSelect(i, SELECT_BY_POS) &&
         OrderMagicNumber() == MagicNumber &&
         OrderSymbol() == SymbolToTrade)
         return(true);
   return(false);
}

void OnTick()
{
   if(HasOpenOrders())
      return;

   double prob = GetProbability();
   if(EnableDebugLogging)
   {
      string feat_vals = "";
      int n_feats = ArraySize(ModelCoefficients);
      for(int i=0; i<n_feats; i++)
      {
         if(i>0) feat_vals += ",";
         feat_vals += DoubleToString(GetFeature(i), 2);
      }
      Print("Features: [" + feat_vals + "] prob=" + DoubleToString(prob, 4));
   }

   // Open buy if probability exceeds threshold else sell
   int ticket;
   if(prob > ModelThreshold)
   {
      ticket = OrderSend(SymbolToTrade, OP_BUY, Lots, Ask, 3, 0, 0,
                         "model", MagicNumber, 0, clrBlue);
   }
   else
   {
      ticket = OrderSend(SymbolToTrade, OP_SELL, Lots, Bid, 3, 0, 0,
                         "model", MagicNumber, 0, clrRed);
   }

   if(ticket < 0)
      Print("OrderSend error: ", GetLastError());
}

void OnDeinit(const int reason)
{
}
