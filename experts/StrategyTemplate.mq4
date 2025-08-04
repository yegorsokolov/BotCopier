#property strict
#include "model_interface.mqh"

extern string SymbolToTrade = "EURUSD";
extern double Lots = 0.1;
extern int MagicNumber = 1234;
extern bool EnableDebugLogging = false;
extern double MinLots = 0.01;
extern double MaxLots = 0.1;
extern string ModelFileName = "model.json";
extern int ReloadModelInterval = 0; // seconds, 0=disabled
extern double BreakEvenPips = 0;
extern double TrailingPips = 0;

int ModelCount = __MODEL_COUNT__;
double ModelCoefficients[__MODEL_COUNT__][__FEATURE_COUNT__] = {__COEFFICIENTS__};
double ModelIntercepts[] = {__INTERCEPTS__};
double CalibrationCoef = __CAL_COEF__;
double CalibrationIntercept = __CAL_INTERCEPT__;
double ModelThreshold = __THRESHOLD__;
double HourlyThresholds[] = {__HOURLY_THRESHOLDS__};
double ProbabilityLookup[] = {__PROBABILITY_TABLE__};
double SLModelCoefficients[] = {__SL_COEFFICIENTS__};
double SLModelIntercept = __SL_INTERCEPT__;
double TPModelCoefficients[] = {__TP_COEFFICIENTS__};
double TPModelIntercept = __TP_INTERCEPT__;
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
double TransformerQKernel[] = {__TRANS_QK__};
double TransformerQBias[] = {__TRANS_QB__};
double TransformerKKernel[] = {__TRANS_KK__};
double TransformerKBias[] = {__TRANS_KB__};
double TransformerVKernel[] = {__TRANS_VK__};
double TransformerVBias[] = {__TRANS_VB__};
double TransformerOutKernel[] = {__TRANS_OK__};
double TransformerOutBias[] = {__TRANS_OB__};
double TransformerDenseWeights[] = {__TRANS_DENSE_W__};
double TransformerDenseBias = __TRANS_DENSE_B__;
double FeatureMean[] = {__FEATURE_MEAN__};
double FeatureStd[] = {__FEATURE_STD__};
datetime CalendarTimes[] = {__CALENDAR_TIMES__};
double CalendarImpacts[] = {__CALENDAR_IMPACTS__};
int EventWindowMinutes = __EVENT_WINDOW__;
double FeatureHistory[__LSTM_SEQ_LEN__][__FEATURE_COUNT__];
int FeatureHistorySize = 0;
int CachedTimeframes[] = {__CACHE_TIMEFRAMES__};
datetime CachedBarTimes[__CACHE_TF_COUNT__];
double CachedSMA[__CACHE_TF_COUNT__];
double CachedRSI[__CACHE_TF_COUNT__];
double CachedMACD[__CACHE_TF_COUNT__];
double CachedMACDSignal[__CACHE_TF_COUNT__];
int LastCalcPeriod = 0;
double CachedBookBidVol = 0.0;
double CachedBookAskVol = 0.0;
double CachedBookImbalance = 0.0;
datetime CachedBookTime = 0;
int EncoderWindow = __ENCODER_WINDOW__;
int EncoderDim = __ENCODER_DIM__;
double EncoderWeights[] = {__ENCODER_WEIGHTS__};
int EncoderCenterCount = __ENCODER_CENTER_COUNT__;
double EncoderCenters[] = {__ENCODER_CENTERS__};
datetime LastModelLoad = 0;

//----------------------------------------------------------------------
// Model loading utilities
//----------------------------------------------------------------------

double ExtractJsonNumber(string json, string key)
{
   int pos = StringFind(json, key);
   if(pos < 0)
      return(0.0);
   pos = StringFind(json, ":", pos);
   if(pos < 0)
      return(0.0);
   pos++;
   while(pos < StringLen(json) && StringMid(json, pos, 1) == " ") pos++;
   int end = pos;
   while(end < StringLen(json))
   {
      string ch = StringMid(json, end, 1);
      if(ch == "," || ch == "}")
         break;
      end++;
   }
   string val = StringSubstr(json, pos, end-pos);
   return(StrToDouble(val));
}

void ExtractJsonArray(string json, string key, double &arr[])
{
   int pos = StringFind(json, key);
   if(pos < 0) return;
   pos = StringFind(json, "[", pos);
   if(pos < 0) return;
   int end = StringFind(json, "]", pos);
   if(end < 0) return;
   string vals = StringSubstr(json, pos+1, end-pos-1);
   string parts[];
   int cnt = StringSplit(vals, ',', parts);
   ArrayResize(arr, cnt);
   for(int i=0; i<cnt; i++)
      arr[i] = StrToDouble(StringTrimLeft(StringTrimRight(parts[i])));
}

bool ParseModelJson(string json)
{
   double tmp[];
   ExtractJsonArray(json, "\"coefficients\"", tmp);
   int n = MathMin(ArraySize(tmp), ArrayRange(ModelCoefficients,1));
   for(int i=0;i<n;i++)
      ModelCoefficients[0][i] = tmp[i];
   ExtractJsonArray(json, "\"hourly_thresholds\"", HourlyThresholds);
   ExtractJsonArray(json, "\"probability_table\"", ProbabilityLookup);
   ExtractJsonArray(json, "\"sl_coefficients\"", SLModelCoefficients);
   ExtractJsonArray(json, "\"tp_coefficients\"", TPModelCoefficients);
   ExtractJsonArray(json, "\"mean\"", FeatureMean);
   if(ArraySize(FeatureMean)==0)
      ExtractJsonArray(json, "\"feature_mean\"", FeatureMean);
   ExtractJsonArray(json, "\"std\"", FeatureStd);
   if(ArraySize(FeatureStd)==0)
      ExtractJsonArray(json, "\"feature_std\"", FeatureStd);
  if(ArraySize(ModelIntercepts)>0)
      ModelIntercepts[0] = ExtractJsonNumber(json, "\"intercept\"");
   CalibrationCoef = ExtractJsonNumber(json, "\"calibration_coef\"");
   if(CalibrationCoef==0) CalibrationCoef = 1;
   CalibrationIntercept = ExtractJsonNumber(json, "\"calibration_intercept\"");
   SLModelIntercept = ExtractJsonNumber(json, "\"sl_intercept\"");
   TPModelIntercept = ExtractJsonNumber(json, "\"tp_intercept\"");
   ModelThreshold = ExtractJsonNumber(json, "\"threshold\"");
   ModelCount = 1;
   return(true);
}

bool ParseModelCsv(string line)
{
   string parts[];
   int cnt = StringSplit(StringTrimLeft(StringTrimRight(line)), ',', parts);
   if(cnt < 3)
      return(false);
   if(ArraySize(ModelIntercepts)>0)
      ModelIntercepts[0] = StrToDouble(parts[0]);
   ModelThreshold = StrToDouble(parts[1]);
   int n = MathMin(cnt - 2, ArrayRange(ModelCoefficients,1));
   for(int i=0; i<n; i++)
      ModelCoefficients[0][i] = StrToDouble(StringTrimLeft(StringTrimRight(parts[i+2])));
   ModelCount = 1;
   return(true);
}

bool LoadModel()
{
   int h = FileOpen(ModelFileName, FILE_READ|FILE_TXT|FILE_COMMON);
   if(h == INVALID_HANDLE)
   {
      Print("Model load failed: ", GetLastError());
      return(false);
   }
   string content = "";
   while(!FileIsEnding(h))
      content += FileReadString(h);
   FileClose(h);
   if(StringFind(StringToLower(ModelFileName), ".json") >= 0)
      return(ParseModelJson(content));
   else
      return(ParseModelCsv(content));
}

int OnInit()
{
   bool ok = LoadModel();
   LastModelLoad = TimeCurrent();
   if(!ok)
      Print("Using built-in model parameters");
   MarketBookAdd(SymbolToTrade);
   return(INIT_SUCCEEDED);
}

//----------------------------------------------------------------------
// Feature extraction utilities
//----------------------------------------------------------------------

double HourSin()
{
   return(MathSin(2.0 * 3.141592653589793 * TimeHour(TimeCurrent()) / 24.0));
}

double HourCos()
{
   return(MathCos(2.0 * 3.141592653589793 * TimeHour(TimeCurrent()) / 24.0));
}

double DowSin()
{
   return(MathSin(2.0 * 3.141592653589793 * TimeDayOfWeek(TimeCurrent()) / 7.0));
}

double DowCos()
{
   return(MathCos(2.0 * 3.141592653589793 * TimeDayOfWeek(TimeCurrent()) / 7.0));
}

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

double GetCalendarFlag()
{
   datetime now = TimeCurrent();
   for(int i=0; i<ArraySize(CalendarTimes); i++)
      if(MathAbs(now - CalendarTimes[i]) <= EventWindowMinutes * 60)
         return(1.0);
   return(0.0);
}

double GetCalendarImpact()
{
   datetime now = TimeCurrent();
   double maxImp = 0.0;
   for(int i=0; i<ArraySize(CalendarTimes); i++)
      if(MathAbs(now - CalendarTimes[i]) <= EventWindowMinutes * 60)
         if(CalendarImpacts[i] > maxImp)
            maxImp = CalendarImpacts[i];
   return(maxImp);
}

double GetEncodedFeature(int idx)
{
   if(idx >= EncoderDim)
      return(0.0);
   double val = 0.0;
   int base = idx * EncoderWindow;
   for(int i=0; i<EncoderWindow; i++)
   {
      double diff = iClose(SymbolToTrade, 0, i) - iClose(SymbolToTrade, 0, i+1);
      val += EncoderWeights[base + i] * diff;
   }
   return(val);
}

int GetRegime()
{
   if(EncoderCenterCount <= 0)
      return(0);
   double enc[100];
   for(int i=0; i<EncoderDim && i<100; i++)
      enc[i] = GetEncodedFeature(i);
   int best = 0;
   double bestDist = 0.0;
   for(int c=0; c<EncoderCenterCount; c++)
   {
      double d = 0.0;
      int base = c * EncoderDim;
      for(int j=0; j<EncoderDim; j++)
      {
         double diff = enc[j] - EncoderCenters[base + j];
         d += diff * diff;
      }
      if(c == 0 || d < bestDist)
      {
         bestDist = d;
         best = c;
      }
   }
   return(best);
}

int TFIdx(int tf)
{
   for(int i=0; i<ArraySize(CachedTimeframes); i++)
      if(CachedTimeframes[i] == tf)
         return(i);
   return(0);
}

void RefreshIndicatorCache()
{
   int curPeriod = Period();
   if(curPeriod != LastCalcPeriod)
   {
      for(int i=0; i<ArraySize(CachedTimeframes); i++)
         if(CachedTimeframes[i] == 0)
            CachedBarTimes[i] = 0;
      LastCalcPeriod = curPeriod;
   }
   for(int i=0; i<ArraySize(CachedTimeframes); i++)
   {
      int tf = CachedTimeframes[i];
      int actual_tf = (tf == 0) ? curPeriod : tf;
      datetime t = iTime(SymbolToTrade, actual_tf, 0);
      if(t != CachedBarTimes[i])
      {
         CachedBarTimes[i] = t;
         CachedSMA[i] = iMA(SymbolToTrade, actual_tf, 5, 0, MODE_SMA, PRICE_CLOSE, 0);
         CachedRSI[i] = iRSI(SymbolToTrade, actual_tf, 14, PRICE_CLOSE, 0);
         CachedMACD[i] = iMACD(SymbolToTrade, actual_tf, 12, 26, 9, PRICE_CLOSE, MODE_MAIN, 0);
         CachedMACDSignal[i] = iMACD(SymbolToTrade, actual_tf, 12, 26, 9, PRICE_CLOSE, MODE_SIGNAL, 0);
      }
   }
}

void RefreshBookCache()
{
   datetime t = TimeCurrent();
   if(t != CachedBookTime)
   {
      MqlBookInfo book[];
      double bid = 0.0;
      double ask = 0.0;
      if(MarketBookGet(SymbolToTrade, book))
      {
         for(int i=0; i<ArraySize(book); i++)
         {
            if(book[i].type==BOOK_TYPE_BUY)
               bid += book[i].volume;
            else if(book[i].type==BOOK_TYPE_SELL)
               ask += book[i].volume;
         }
      }
      CachedBookBidVol = bid;
      CachedBookAskVol = ask;
      CachedBookImbalance = (bid+ask>0) ? (bid-ask)/(bid+ask) : 0.0;
      CachedBookTime = t;
   }
}

double BookBidVol()
{
   RefreshBookCache();
   return(CachedBookBidVol);
}

double BookAskVol()
{
   RefreshBookCache();
   return(CachedBookAskVol);
}

double BookImbalance()
{
   RefreshBookCache();
   return(CachedBookImbalance);
}

double PairCorrelation(string sym1, string sym2, int window=5)
{
   double mean1 = iMA(sym1, 0, window, 0, MODE_SMA, PRICE_CLOSE, 1);
   double mean2 = iMA(sym2, 0, window, 0, MODE_SMA, PRICE_CLOSE, 1);
   double num = 0.0;
   double den1 = 0.0;
   double den2 = 0.0;
   for(int i=1; i<=window; i++)
   {
      double d1 = iClose(sym1, 0, i) - mean1;
      double d2 = iClose(sym2, 0, i) - mean2;
      num += d1 * d2;
      den1 += d1 * d1;
      den2 += d2 * d2;
   }
   if(den1 <= 0 || den2 <= 0)
      return(0.0);
   return(num / MathSqrt(den1 * den2));
}

double GetFeature(int index)
{
   /* Return a simple set of features for the logistic model.
      Feature mapping is intentionally minimal and should be kept in
      sync with the Python training script.  Unknown indices default
      to zero. */
   RefreshIndicatorCache();
   double raw = 0.0;
   switch(index)
   {
__FEATURE_CASES__      default:
         raw = 0.0;
         break;
   }
   if(index < ArraySize(FeatureMean) && index < ArraySize(FeatureStd) && FeatureStd[index] != 0)
      return((raw - FeatureMean[index]) / FeatureStd[index]);
   return(raw);
}

double ComputeLogisticScore()
{
   double total = 0.0;
   for(int m=0; m<ModelCount; m++)
   {
      double z = ModelIntercepts[m];
      for(int i=0; i<FeatureCount; i++)
         z += ModelCoefficients[m][i] * GetFeature(i);
      z = CalibrationCoef*z + CalibrationIntercept;
      total += 1.0 / (1.0 + MathExp(-z));
   }
   return(total/ModelCount);
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

double ComputeTransformerScore()
{
   int steps = FeatureHistorySize;
   if(steps > LSTMSequenceLength)
      steps = LSTMSequenceLength;
   int f = FeatureCount;
   double Q[100][100];
   double K[100][100];
   double V[100][100];
   for(int t=0; t<steps; t++)
      for(int j=0; j<f; j++)
      {
         double xq = TransformerQBias[j];
         double xk = TransformerKBias[j];
         double xv = TransformerVBias[j];
         for(int i=0; i<f; i++)
         {
            double x = FeatureHistory[steps-1 - t][i];
            xq += x * TransformerQKernel[i*f + j];
            xk += x * TransformerKKernel[i*f + j];
            xv += x * TransformerVKernel[i*f + j];
         }
         Q[t][j] = xq;
         K[t][j] = xk;
         V[t][j] = xv;
      }
   double Context[100][100];
   ArrayInitialize(Context, 0.0);
   for(int i=0; i<steps; i++)
   {
      double scores[100];
      double denom = 0.0;
      for(int j=0; j<steps; j++)
      {
         double s = 0.0;
         for(int d=0; d<f; d++)
            s += Q[i][d] * K[j][d];
         s = MathExp(s / MathSqrt(f));
         scores[j] = s;
         denom += s;
      }
      for(int j=0; j<steps; j++)
      {
         double w = scores[j] / denom;
         for(int d=0; d<f; d++)
            Context[i][d] += w * V[j][d];
      }
   }
   double Out[100][100];
   for(int i=0; i<steps; i++)
      for(int d=0; d<f; d++)
      {
         double s = TransformerOutBias[d];
         for(int j=0; j<f; j++)
            s += Context[i][j] * TransformerOutKernel[j*f + d];
         Out[i][d] = s;
      }
   double pooled[100];
   for(int d=0; d<f; d++)
   {
      double sum = 0.0;
      for(int i=0; i<steps; i++)
         sum += Out[i][d];
      pooled[d] = sum / steps;
   }
   double z = TransformerDenseBias;
   for(int d=0; d<f; d++)
      z += TransformerDenseWeights[d] * pooled[d];
   return(1.0 / (1.0 + MathExp(-z)));
}

double GetProbability()
{
   UpdateFeatureHistory();
   if(ArraySize(ProbabilityLookup) == 24)
      return(ProbabilityLookup[TimeHour(TimeCurrent())]);
   if(LSTMSequenceLength > 0 && ArraySize(TransformerDenseWeights) > 0)
      return(ComputeTransformerScore());
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

double PredictSLDistance()
{
   double z = SLModelIntercept;
   int n = ArraySize(SLModelCoefficients);
   for(int i=0; i<n; i++)
      z += SLModelCoefficients[i] * GetFeature(i);
   return(z);
}

double PredictTPDistance()
{
   double z = TPModelIntercept;
   int n = ArraySize(TPModelCoefficients);
   for(int i=0; i<n; i++)
      z += TPModelCoefficients[i] * GetFeature(i);
   return(z);
}

double GetNewSL(bool isBuy)
{
   double d = PredictSLDistance();
   if(d <= 0) return(0);
   if(isBuy) return(Bid - d);
   return(Ask + d);
}

double GetNewTP(bool isBuy)
{
   double d = PredictTPDistance();
   if(d <= 0) return(0);
   if(isBuy) return(Bid + d);
   return(Ask - d);
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
   if(ReloadModelInterval > 0 && TimeCurrent() - LastModelLoad >= ReloadModelInterval)
   {
      if(LoadModel())
         Print("Model parameters reloaded");
      LastModelLoad = TimeCurrent();
   }
   if(HasOpenOrders())
   {
      ManageOpenOrders();
      return;
   }

   double feats[100];
   for(int i=0; i<FeatureCount && i<100; i++)
      feats[i] = GetFeature(i);

   double prob_sum = 0.0;
   for(int m=0; m<ModelCount; m++)
   {
      double z = ModelIntercepts[m];
      for(int i=0; i<FeatureCount; i++)
         z += ModelCoefficients[m][i] * feats[i];
      prob_sum += 1.0 / (1.0 + MathExp(-z));
   }
   double prob = prob_sum / MathMax(ModelCount,1);

   if(EnableDebugLogging)
   {
      string feat_vals = "";
      for(int i=0; i<FeatureCount; i++)
      {
         if(i>0) feat_vals += ",";
         feat_vals += DoubleToString(feats[i], 2);
      }
      Print("Features: [" + feat_vals + "] prob=" + DoubleToString(prob, 4));
   }

   // Open buy if probability exceeds threshold else sell
   double tradeLots = GetTradeLots(prob);
   int ticket;
   double thr = GetTradeThreshold();
   if(prob > thr)
   {
      ticket = OrderSend(SymbolToTrade, OP_BUY, tradeLots, Ask, 3,
                         GetNewSL(true), GetNewTP(true),
                         "model", MagicNumber, 0, clrBlue);
   }
   else
   {
      ticket = OrderSend(SymbolToTrade, OP_SELL, tradeLots, Bid, 3,
                         GetNewSL(false), GetNewTP(false),
                         "model", MagicNumber, 0, clrRed);
   }

   if(ticket < 0)
      Print("OrderSend error: ", GetLastError());
}

void ManageOpenOrders()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(OrderMagicNumber() != MagicNumber || OrderSymbol() != SymbolToTrade)
         continue;

      bool isBuy = (OrderType() == OP_BUY);
      double price = isBuy ? Bid : Ask;
      double profitPips = (isBuy ? (price - OrderOpenPrice()) : (OrderOpenPrice() - price)) / Point;

      double newSL = OrderStopLoss();
      if(BreakEvenPips > 0 && profitPips >= BreakEvenPips)
      {
         double breakeven = OrderOpenPrice();
         if(isBuy && (OrderStopLoss() < breakeven || OrderStopLoss() == 0))
            newSL = breakeven;
         if(!isBuy && (OrderStopLoss() > breakeven || OrderStopLoss() == 0))
            newSL = breakeven;
      }

      if(TrailingPips > 0 && profitPips > TrailingPips)
      {
         double trail = isBuy ? price - TrailingPips * Point : price + TrailingPips * Point;
         if(isBuy && trail > newSL)
            newSL = trail;
         if(!isBuy && (trail < newSL || newSL == 0))
            newSL = trail;
      }

      if(newSL != OrderStopLoss())
      {
         if(!OrderModify(OrderTicket(), OrderOpenPrice(), NormalizeDouble(newSL, Digits), OrderTakeProfit(), 0, clrYellow))
            Print("OrderModify error: ", GetLastError());
      }
   }
}

void OnDeinit(const int reason)
{
   MarketBookRelease(SymbolToTrade);
}
