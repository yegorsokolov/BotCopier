#property strict
#include "model_interface.mqh"

extern string SymbolToTrade = "EURUSD";
extern double Lots = 0.1;
extern int MagicNumber = 1234;
extern bool EnableDebugLogging = false;
extern double MinLots = 0.01;
extern double MaxLots = 0.1;

double ModelCoefficients[] = {__COEFFICIENTS__};
double ModelIntercept = __INTERCEPT__;
double ModelThreshold = __THRESHOLD__;
double HourlyThresholds[] = {__HOURLY_THRESHOLDS__};
double ProbabilityLookup[] = {__PROBABILITY_TABLE__};
int ModelHiddenSize = __NN_HIDDEN_SIZE__;
double NNLayer1Weights[] = {__NN_L1_WEIGHTS__};
double NNLayer1Bias[] = {__NN_L1_BIAS__};
double NNLayer2Weights[] = {__NN_L2_WEIGHTS__};
double NNLayer2Bias = __NN_L2_BIAS__;
int LSTMSequenceLength = __LSTM_SEQ_LEN__;
int LSTMHiddenSize = __LSTM_HIDDEN_SIZE__;
int FeatureCount = __FEATURE_COUNT__;
double LSTMKernels[] = {__LSTM_KERNEL__};
double LSTMRecurrent[] = {__LSTM_RECURRENT__};
double LSTMBias[] = {__LSTM_BIAS__};
double LSTMDenseWeights[] = {__LSTM_DENSE_W__};
double LSTMDenseBias = __LSTM_DENSE_B__;
double FeatureHistory[__LSTM_SEQ_LEN__][__FEATURE_COUNT__];
int FeatureHistorySize = 0;

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

double ComputeNNScore()
{
   int hidden = ModelHiddenSize;
   if(hidden <= 0)
      return(ComputeLogisticScore());
   int inputCount = ArraySize(NNLayer1Weights) / hidden;
   double z = NNLayer2Bias;
   for(int j=0; j<hidden; j++)
   {
      double h = NNLayer1Bias[j];
      for(int i=0; i<inputCount; i++)
         h += NNLayer1Weights[j*inputCount + i] * GetFeature(i);
      if(h < 0) h = 0; // ReLU
      z += NNLayer2Weights[j] * h;
   }
   return(1.0 / (1.0 + MathExp(-z)));
}

void UpdateFeatureHistory()
{
   for(int j=LSTMSequenceLength-1; j>0; j--)
      for(int i=0; i<FeatureCount; i++)
         FeatureHistory[j][i] = FeatureHistory[j-1][i];
   for(int i=0; i<FeatureCount; i++)
      FeatureHistory[0][i] = GetFeature(i);
   if(FeatureHistorySize < LSTMSequenceLength)
      FeatureHistorySize++;
}

double ComputeLSTMScore()
{
   int hidden = LSTMHiddenSize;
   if(hidden <= 0)
      return(ComputeNNScore());
   double h[100];
   double c[100];
   ArrayInitialize(h, 0.0);
   ArrayInitialize(c, 0.0);
   int steps = FeatureHistorySize;
   if(steps > LSTMSequenceLength)
      steps = LSTMSequenceLength;
   for(int t=steps-1; t>=0; t--)
   {
      for(int u=0; u<hidden; u++)
      {
         double zi = LSTMBias[u];
         double zf = LSTMBias[hidden + u];
         double zg = LSTMBias[hidden*2 + u];
         double zo = LSTMBias[hidden*3 + u];
         for(int i=0; i<FeatureCount; i++)
         {
            int base = i*hidden*4;
            double x = FeatureHistory[t][i];
            zi += x * LSTMKernels[base + u];
            zf += x * LSTMKernels[base + hidden + u];
            zg += x * LSTMKernels[base + hidden*2 + u];
            zo += x * LSTMKernels[base + hidden*3 + u];
         }
         for(int k=0; k<hidden; k++)
         {
            int base = k*hidden*4;
            double hp = h[k];
            zi += hp * LSTMRecurrent[base + u];
            zf += hp * LSTMRecurrent[base + hidden + u];
            zg += hp * LSTMRecurrent[base + hidden*2 + u];
            zo += hp * LSTMRecurrent[base + hidden*3 + u];
         }
         double i_g = 1.0 / (1.0 + MathExp(-zi));
         double f_g = 1.0 / (1.0 + MathExp(-zf));
         double g_g = MathTanH(zg);
         double o_g = 1.0 / (1.0 + MathExp(-zo));
         c[u] = f_g * c[u] + i_g * g_g;
         h[u] = o_g * MathTanH(c[u]);
      }
   }
   double z = LSTMDenseBias;
   for(int u=0; u<hidden; u++)
      z += LSTMDenseWeights[u] * h[u];
   return(1.0 / (1.0 + MathExp(-z)));
}

double GetProbability()
{
   UpdateFeatureHistory();
   if(ArraySize(ProbabilityLookup) == 24)
      return(ProbabilityLookup[TimeHour(TimeCurrent())]);
   if(LSTMSequenceLength > 0 && ArraySize(LSTMDenseWeights) > 0)
      return(ComputeLSTMScore());
   if(ArraySize(NNLayer1Weights) > 0)
      return(ComputeNNScore());
   return(ComputeLogisticScore());
}

double GetTradeLots(double prob)
{
   double x = 1.0 / (1.0 + MathExp(-10.0*(prob - 0.5)));
   double lots = MinLots + (MaxLots - MinLots) * x;
   return(lots);
}

double GetTradeThreshold()
{
   if(ArraySize(HourlyThresholds) == 24)
      return(HourlyThresholds[TimeHour(TimeCurrent())]);
   return(ModelThreshold);
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
      if(n_feats == 0 && ModelHiddenSize > 0)
         n_feats = ArraySize(NNLayer1Weights) / ModelHiddenSize;
      for(int i=0; i<n_feats; i++)
      {
         if(i>0) feat_vals += ",";
         feat_vals += DoubleToString(GetFeature(i), 2);
      }
      Print("Features: [" + feat_vals + "] prob=" + DoubleToString(prob, 4));
   }

   // Open buy if probability exceeds threshold else sell
   double tradeLots = GetTradeLots(prob);
   int ticket;
   double thr = GetTradeThreshold();
   if(prob > thr)
   {
      ticket = OrderSend(SymbolToTrade, OP_BUY, tradeLots, Ask, 3, 0, 0,
                         "model", MagicNumber, 0, clrBlue);
   }
   else
   {
      ticket = OrderSend(SymbolToTrade, OP_SELL, tradeLots, Bid, 3, 0, 0,
                         "model", MagicNumber, 0, clrRed);
   }

   if(ticket < 0)
      Print("OrderSend error: ", GetLastError());
}

void OnDeinit(const int reason)
{
}
