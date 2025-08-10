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
extern bool   EnableDecisionLogging = true;
extern bool   DecisionLogToSocket = false;
extern string DecisionLogFile = "decisions.csv";
extern string UncertainDecisionFile = "uncertain_decisions.csv";
extern double UncertaintyMargin = 0.05;
extern string DecisionLogSocketHost = "127.0.0.1";
extern int    DecisionLogSocketPort = 9001;
extern string ModelVersion = "";
extern string EncoderOnnxFile = "__ENCODER_ONNX__";

string HierarchyJson = "__HIERARCHY_JSON__";

int ModelCount = __MODEL_COUNT__;
int SessionStarts[] = {__SESSION_STARTS__};
int SessionEnds[] = {__SESSION_ENDS__};
double ModelCoefficients[__MODEL_COUNT__][__FEATURE_COUNT__] = {__COEFFICIENTS__};
double ModelIntercepts[] = {__INTERCEPTS__};
double ModelCoeffVar[__MODEL_COUNT__][__FEATURE_COUNT__] = {__COEF_VARIANCES__};
double ModelNoiseVar[] = {__NOISE_VARIANCES__};
double GatingCoefficients[__MODEL_COUNT__][__FEATURE_COUNT__] = {__GATING_COEFFICIENTS__};
double GatingIntercepts[] = {__GATING_INTERCEPTS__};
double CalibrationCoef = __CAL_COEF__;
double CalibrationIntercept = __CAL_INTERCEPT__;
double ModelThreshold = __THRESHOLD__;
double HourlyThresholds[] = {__HOURLY_THRESHOLDS__};
double ProbabilityLookup[__MODEL_COUNT__][24] = {__PROBABILITY_TABLE__};
double SLModelCoefficients[] = {__SL_COEFFICIENTS__};
double SLModelIntercept = __SL_INTERCEPT__;
double TPModelCoefficients[] = {__TP_COEFFICIENTS__};
double TPModelIntercept = __TP_INTERCEPT__;
double LotModelCoefficients[] = {__LOT_COEFFICIENTS__};
double LotModelIntercept = __LOT_INTERCEPT__;
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
double CachedNewsSentiment = 0.0;
datetime CachedNewsTime = 0;
int EncoderWindow = __ENCODER_WINDOW__;
int EncoderDim = __ENCODER_DIM__;
double EncoderWeights[] = {__ENCODER_WEIGHTS__};
int EncoderCenterCount = __ENCODER_CENTER_COUNT__;
double EncoderCenters[] = {__ENCODER_CENTERS__};
int RegimeCount = __REGIME_COUNT__;
int RegimeFeatureCount = __REGIME_FEATURE_COUNT__;
double RegimeCenters[__REGIME_COUNT__][__REGIME_FEATURE_COUNT__] = {__REGIME_CENTERS__};
int RegimeFeatureIdx[] = {__REGIME_FEATURE_IDX__};
datetime LastModelLoad = 0;
int      DecisionLogHandle = INVALID_HANDLE;
int      UncertainLogHandle = INVALID_HANDLE;
int      DecisionSocket = INVALID_HANDLE;
int      NextDecisionId = 1;
bool UseOnnxEncoder = false;

#import "onnxruntime_wrapper.ex4"
   int OnnxEncode(string model, double &inp[], double &out[]);
#import

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
   double prob_tmp[];
   ExtractJsonArray(json, "\"probability_table\"", prob_tmp);
   if(ArraySize(prob_tmp) == 24 && ArrayRange(ProbabilityLookup,0) > 0)
      for(int i=0;i<24;i++)
         ProbabilityLookup[0][i] = prob_tmp[i];
   ExtractJsonArray(json, "\"sl_coefficients\"", SLModelCoefficients);
   ExtractJsonArray(json, "\"tp_coefficients\"", TPModelCoefficients);
   ExtractJsonArray(json, "\"lot_coefficients\"", LotModelCoefficients);
   ExtractJsonArray(json, "\"coef_variances\"", tmp);
   n = MathMin(ArraySize(tmp), ArrayRange(ModelCoeffVar,1));
   for(int i=0;i<n;i++)
      ModelCoeffVar[0][i] = tmp[i];
   ModelNoiseVar[0] = ExtractJsonNumber(json, "\"noise_variance\"");
   ExtractJsonArray(json, "\"gating_coefficients\"", tmp);
   int gtot = MathMin(ArraySize(tmp), ArrayRange(GatingCoefficients,0)*ArrayRange(GatingCoefficients,1));
   for(int i=0;i<gtot;i++)
      GatingCoefficients[i/ArrayRange(GatingCoefficients,1)][i%ArrayRange(GatingCoefficients,1)] = tmp[i];
   ExtractJsonArray(json, "\"gating_intercepts\"", GatingIntercepts);
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
   LotModelIntercept = ExtractJsonNumber(json, "\"lot_intercept\"");
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
   string lower = StringToLower(ModelFileName);
   bool is_gz = StringFind(lower, ".gz") >= 0;
   int mode = FILE_READ|FILE_COMMON|(is_gz ? FILE_BIN : FILE_TXT);
   int h = FileOpen(ModelFileName, mode);
   if(h == INVALID_HANDLE)
   {
      Print("Model load failed: ", GetLastError());
      return(false);
   }
   string content = "";
   if(is_gz)
   {
      int size = FileSize(h);
      uchar data[];
      ArrayResize(data, size);
      FileReadArray(h, data, 0, size);
      FileClose(h);
      uchar raw[];
      if(!CryptDecode(CRYPT_ARCHIVE_GZIP, data, raw))
      {
         Print("Gzip decode failed");
         return(false);
      }
      content = CharArrayToString(raw, 0, WHOLE_ARRAY, CP_UTF8);
   }
   else
   {
      while(!FileIsEnding(h))
         content += FileReadString(h);
      FileClose(h);
   }
   if(StringFind(lower, ".json") >= 0)
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
   if(StringLen(EncoderOnnxFile) > 0 && FileIsExist(EncoderOnnxFile))
      UseOnnxEncoder = true;
   MarketBookAdd(SymbolToTrade);
   if(EnableDecisionLogging)
   {
      if(DecisionLogToSocket)
      {
         DecisionSocket = SocketCreate();
         if(DecisionSocket != INVALID_HANDLE)
         {
            if(!SocketConnect(DecisionSocket, DecisionLogSocketHost, DecisionLogSocketPort, 1000))
            {
               Print("Decision log socket connect failed: ", GetLastError());
               SocketClose(DecisionSocket);
               DecisionSocket = INVALID_HANDLE;
            }
         }
         else
            Print("Decision log socket creation failed: ", GetLastError());
      }
      else
      {
         DecisionLogHandle = FileOpen(DecisionLogFile, FILE_CSV|FILE_WRITE|FILE_READ|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ, ';');
         if(DecisionLogHandle != INVALID_HANDLE)
         {
            if(FileSize(DecisionLogHandle) == 0)
               FileWrite(DecisionLogHandle, "event_id;timestamp;model_version;action;probability;features");
            FileSeek(DecisionLogHandle, 0, SEEK_END);
         }
         else
            Print("Decision log open failed: ", GetLastError());
      }
  }
   UncertainLogHandle = FileOpen(UncertainDecisionFile, FILE_CSV|FILE_WRITE|FILE_READ|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ, ';');
   if(UncertainLogHandle != INVALID_HANDLE)
   {
      if(FileSize(UncertainLogHandle) == 0)
         FileWrite(UncertainLogHandle, "event_id;timestamp;model_version;action;probability;sl_dist;tp_dist;features");
      FileSeek(UncertainLogHandle, 0, SEEK_END);
   }
   else
      Print("Uncertain decision log open failed: ", GetLastError());
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
   if(UseOnnxEncoder)
   {
      double seq[];
      ArrayResize(seq, EncoderWindow);
      for(int i=0; i<EncoderWindow; i++)
         seq[i] = iClose(SymbolToTrade,0,i) - iClose(SymbolToTrade,0,i+1);
      double out[];
      ArrayResize(out, EncoderDim);
      if(OnnxEncode(EncoderOnnxFile, seq, out) == 0 && idx < ArraySize(out))
         return(out[idx]);
   }
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
   double feats[100];
   for(int i=0; i<RegimeFeatureCount && i<100; i++)
      feats[i] = GetFeature(RegimeFeatureIdx[i]);
   int best = 0;
   double bestDist = 0.0;
   for(int c=0; c<RegimeCount; c++)
   {
      double d = 0.0;
      for(int j=0; j<RegimeFeatureCount; j++)
      {
         double diff = feats[j] - RegimeCenters[c][j];
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

double PairCorrelation(string sym1, string sym2="", int window=5)
{
   if(sym2 == "")
   {
      sym2 = sym1;
      sym1 = SymbolToTrade;
   }
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

double GetNewsSentiment()
{
   if(TimeCurrent() == CachedNewsTime)
      return(CachedNewsSentiment);
   CachedNewsTime = TimeCurrent();
   CachedNewsSentiment = 0.0;
   int h = FileOpen("news_sentiment.csv", FILE_READ|FILE_CSV|FILE_COMMON, ';');
   if(h == INVALID_HANDLE)
      return(0.0);
   while(!FileIsEnding(h))
   {
      string ts = FileReadString(h);
      string sym = FileReadString(h);
      double sc = FileReadNumber(h);
      if(sym == SymbolToTrade)
         CachedNewsSentiment = sc;
   }
   FileClose(h);
   return(CachedNewsSentiment);
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
int GetSessionIndex()
{
   int h = TimeHour(TimeCurrent());
   for(int i=0; i<ArraySize(SessionStarts); i++)
      if(h >= SessionStarts[i] && h < SessionEnds[i])
         return(i);
   return(0);
}

int SelectExpert()
{
   int n = ArraySize(GatingIntercepts);
   if(n <= 0)
      return(GetSessionIndex());
   int best = 0;
   double best_z = -1e10;
   for(int m=0; m<n; m++)
   {
      double z = GatingIntercepts[m];
      for(int i=0; i<FeatureCount; i++)
         z += GatingCoefficients[m][i] * GetFeature(i);
      if(z > best_z)
      {
         best_z = z;
         best = m;
      }
   }
   return(best);
}

double ComputeLogisticScoreSession(int m)
{
   double z = ModelIntercepts[m];
   for(int i=0; i<FeatureCount; i++)
      z += ModelCoefficients[m][i] * GetFeature(i);
   z = CalibrationCoef*z + CalibrationIntercept;
   return(1.0 / (1.0 + MathExp(-z)));
}

double ComputePredictiveVariance(int m)
{
   double var = 0.0;
   if(ArraySize(ModelNoiseVar) > m)
      var += ModelNoiseVar[m];
   for(int i=0; i<FeatureCount; i++)
   {
      double f = GetFeature(i);
      var += ModelCoeffVar[m][i] * f * f;
   }
   double z = ModelIntercepts[m];
   for(int i=0; i<FeatureCount; i++)
      z += ModelCoefficients[m][i] * GetFeature(i);
   z = CalibrationCoef*z + CalibrationIntercept;
   double p = 1.0 / (1.0 + MathExp(-z));
   double deriv = p*(1.0 - p);
   return(var * deriv * deriv);
}

double ComputeNNScore()
{
   int hidden = ModelHiddenSize;
   if(hidden <= 0)
      return(ComputeLogisticScoreSession(SelectExpert()));
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

// Compute probability using Decision Transformer weights
double ComputeDecisionTransformerScore()
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
   int sess = SelectExpert();
   if(ArraySize(GatingIntercepts) == 0 && ArrayRange(ProbabilityLookup,0) == ModelCount && ArrayRange(ProbabilityLookup,1) == 24)
      return(ProbabilityLookup[sess][TimeHour(TimeCurrent())]);
   if(LSTMSequenceLength > 0 && ArraySize(TransformerDenseWeights) > 0)
      return(ComputeDecisionTransformerScore());
   if(LSTMSequenceLength > 0 && ArraySize(LSTMDenseWeights) > 0)
      return(ComputeLSTMScore());
   if(ArraySize(NNLayer1Weights) > 0)
      return(ComputeNNScore());
   return(ComputeLogisticScoreSession(sess));
}

double CalcLots()
{
   double z = LotModelIntercept;
   int n = ArraySize(LotModelCoefficients);
   for(int i=0; i<n; i++)
      z += LotModelCoefficients[i] * GetFeature(i);
   int m = SelectExpert();
   double pv = ComputePredictiveVariance(m);
   double scale = 1.0 / (1.0 + pv);
   z *= scale;
   if(z < MinLots) z = MinLots;
   if(z > MaxLots) z = MaxLots;
   return(z);
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

void CalcStops(double &sl_dist, double &tp_dist)
{
   sl_dist = PredictSLDistance();
   tp_dist = PredictTPDistance();
}

void LogDecision(double &feats[], double prob, string action)
{
   if(!EnableDecisionLogging)
      return;
   double sl_dist, tp_dist;
   CalcStops(sl_dist, tp_dist);
   string feat_vals = "";
   for(int i=0; i<FeatureCount; i++)
   {
      if(i>0) feat_vals += ",";
      feat_vals += DoubleToString(feats[i], 5);
   }
   datetime now = TimeCurrent();
   if(DecisionLogHandle != INVALID_HANDLE)
   {
      FileWrite(DecisionLogHandle, NextDecisionId, TimeToString(now, TIME_DATE|TIME_SECONDS), ModelVersion, action, prob, sl_dist, tp_dist, feat_vals);
      FileFlush(DecisionLogHandle);
   }
   bool uncertain = MathAbs(prob - ModelThreshold) <= UncertaintyMargin;
   if(uncertain && UncertainLogHandle != INVALID_HANDLE)
   {
      FileWrite(UncertainLogHandle, NextDecisionId, TimeToString(now, TIME_DATE|TIME_SECONDS), ModelVersion, action, prob, sl_dist, tp_dist, feat_vals);
      FileFlush(UncertainLogHandle);
   }
   if(DecisionSocket != INVALID_HANDLE)
   {
      string json = StringFormat("{\"event_id\":%d,\"timestamp\":\"%s\",\"model_version\":\"%s\",\"action\":\"%s\",\"probability\":%.6f,\"sl_dist\":%.5f,\"tp_dist\":%.5f,\"features\":[%s]}",
                                 NextDecisionId, TimeToString(now, TIME_DATE|TIME_SECONDS), ModelVersion, action, prob, sl_dist, tp_dist, feat_vals);
      uchar bytes[];
      StringToCharArray(json+"\n", bytes, 0, WHOLE_ARRAY, CP_UTF8);
      SocketSend(DecisionSocket, bytes, ArraySize(bytes)-1);
   }
   NextDecisionId++;
}

double GetNewSL(bool isBuy)
{
   double sl_dist, tp_dummy;
   CalcStops(sl_dist, tp_dummy);
   if(sl_dist <= 0) return(0);
   if(isBuy) return(Bid - sl_dist);
   return(Ask + sl_dist);
}

double GetNewTP(bool isBuy)
{
   double sl_dummy, tp_dist;
   CalcStops(sl_dummy, tp_dist);
   if(tp_dist <= 0) return(0);
   if(isBuy) return(Bid + tp_dist);
   return(Ask - tp_dist);
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
   // When ReloadModelInterval is set, periodically check for updated
   // coefficients written by ``online_trainer.py`` and reload them without
   // recompiling the EA.
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

   UpdateFeatureHistory();
   int modelIdx = SelectExpert();
   double prob;
   if(LSTMSequenceLength > 0 && ArraySize(TransformerDenseWeights) > 0)
      prob = ComputeDecisionTransformerScore();
   else if(LSTMSequenceLength > 0 && ArraySize(LSTMDenseWeights) > 0)
      prob = ComputeLSTMScore();
   else if(ArraySize(NNLayer1Weights) > 0)
      prob = ComputeNNScore();
   else
      prob = ComputeLogisticScoreSession(modelIdx);

   double feats[100];
   for(int i=0; i<FeatureCount && i<100; i++)
      feats[i] = GetFeature(i);

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
   double tradeLots = CalcLots();
   int ticket;
   double thr = GetTradeThreshold();
   string action = (prob > thr) ? "buy" : "sell";
   int decision_id = NextDecisionId;
   LogDecision(feats, prob, action);
   string order_comment = StringFormat("model|decision_id=%d", decision_id);
   if(prob > thr)
   {
      ticket = OrderSend(SymbolToTrade, OP_BUY, tradeLots, Ask, 3,
                         GetNewSL(true), GetNewTP(true),
                         order_comment, MagicNumber, 0, clrBlue);
   }
   else
   {
      ticket = OrderSend(SymbolToTrade, OP_SELL, tradeLots, Bid, 3,
                         GetNewSL(false), GetNewTP(false),
                         order_comment, MagicNumber, 0, clrRed);
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
   if(DecisionLogHandle != INVALID_HANDLE)
      FileClose(DecisionLogHandle);
   if(UncertainLogHandle != INVALID_HANDLE)
      FileClose(UncertainLogHandle);
   if(DecisionSocket != INVALID_HANDLE)
      SocketClose(DecisionSocket);
   MarketBookRelease(SymbolToTrade);
}
