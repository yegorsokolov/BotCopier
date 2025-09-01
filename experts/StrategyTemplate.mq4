#property strict
#include "model_interface.mqh"

#import "shm_ring.dll"
bool ShmRingInit(string name, int size);
bool ShmRingRead(int &msg_type, uchar &payload[], int &len);
#import

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
extern bool   ReplayDecisions = false;
extern double UncertaintyMargin = 0.05;
extern double MaxPredictiveVariance = 0.1;
extern string DecisionLogSocketHost = "127.0.0.1";
extern int    DecisionLogSocketPort = 9001;
extern string ModelVersion = "";
extern string EncoderOnnxFile = "__ENCODER_ONNX__";
extern string ModelOnnxFile = "__MODEL_ONNX__";
extern string BanditRouterHost = "127.0.0.1";
extern int    BanditRouterPort = 9100;
extern string MetaAdaptHost = "127.0.0.1";
extern int    MetaAdaptPort = 9200;
extern int    AdaptationInterval = 0; // seconds, 0=disabled
extern string AdaptationLogFile = "adaptations.csv";
extern bool   EnableShadowTrading = false;
extern string ShadowTradesFile = "shadow_trades.csv";

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
double ModelThreshold[] = {__HOURLY_THRESHOLDS__};
double DefaultThreshold = __THRESHOLD__;
double ProbabilityLookup[__MODEL_COUNT__][24] = {__PROBABILITY_TABLE__};
double ConformalLower = __CONFORMAL_LOWER__;
double ConformalUpper = __CONFORMAL_UPPER__;
string ThresholdSymbols[];
double ThresholdTable[][24];
double SLModelCoefficients[] = {__SL_COEFFICIENTS__};
double SLModelIntercept = __SL_INTERCEPT__;
double TPModelCoefficients[] = {__TP_COEFFICIENTS__};
double TPModelIntercept = __TP_INTERCEPT__;
double LotModelCoefficients[] = {__LOT_COEFFICIENTS__};
double LotModelIntercept = __LOT_INTERCEPT__;
double EntryCoefficients[] = {__ENTRY_COEFFICIENTS__};
double EntryIntercept = __ENTRY_INTERCEPT__;
double EntryThreshold = __ENTRY_THRESHOLD__;
double ExitCoefficients[] = {__EXIT_COEFFICIENTS__};
double ExitIntercept = __EXIT_INTERCEPT__;
double ExitThreshold = __EXIT_THRESHOLD__;
string ExitReasonContext = "";
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
int SymbolEmbDim = __SYM_EMB_DIM__;
int SymbolEmbCount = __SYM_EMB_COUNT__;
string SymbolEmbSymbols[] = {__SYM_EMB_SYMBOLS__};
double SymbolEmbeddings[__SYM_EMB_COUNT__][__SYM_EMB_DIM__] = {__SYM_EMB_VALUES__};
string RiskParitySymbols[] = {__RISK_PARITY_SYMBOLS__};
double RiskParityWeights[] = {__RISK_PARITY_WEIGHTS__};
datetime CalendarTimes[] = {__CALENDAR_TIMES__};
double CalendarImpacts[] = {__CALENDAR_IMPACTS__};
int    CalendarIds[] = {__CALENDAR_IDS__};
int EventWindowMinutes = __EVENT_WINDOW__;
int LastCalendarEventId = -1;
// Pre-computed graph metrics injected at build time
string GraphSymbols[] = {__GRAPH_SYMBOLS__};
double GraphDegreeVals[] = {__GRAPH_DEGREE__};
double GraphPagerankVals[] = {__GRAPH_PAGERANK__};
int GraphEmbDim = __GRAPH_EMB_DIM__;
int GraphEmbCount = __GRAPH_EMB_COUNT__;
double GraphEmbeddings[__GRAPH_EMB_COUNT__][__GRAPH_EMB_DIM__] = {__GRAPH_EMB__};
string CointBaseSymbols[] = {__COINT_BASE__};
string CointPeerSymbols[] = {__COINT_PEER__};
double CointBetas[] = {__COINT_BETA__};
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
double CachedBookSpread = 0.0;
double CachedBidAskRatio = 0.0;
double CachedBookImbalanceRoll = 0.0;
double BookImbalanceHist[5];
int BookImbPos = 0;
int BookImbCount = 0;
double CachedNewsSentiment = 0.0;
datetime CachedNewsTime = 0;
double TrendEstimate = 0.0;
double TrendVariance = 1.0;
double LastSlippage = 0.0;
int EncoderWindow = __ENCODER_WINDOW__;
int EncoderDim = __ENCODER_DIM__;
double EncoderWeights[] = {__ENCODER_WEIGHTS__};
int EncoderCenterCount = __ENCODER_CENTER_COUNT__;
double EncoderCenters[] = {__ENCODER_CENTERS__};
int RegimeCount = __REGIME_COUNT__;
int RegimeFeatureCount = __REGIME_FEATURE_COUNT__;
double RegimeCenters[__REGIME_COUNT__][__REGIME_FEATURE_COUNT__] = {__REGIME_CENTERS__};
int RegimeFeatureIdx[] = {__REGIME_FEATURE_IDX__};
const int MSG_REGIME = 3;
int CurrentRegime = -1;
double RegimeThresholds[__REGIME_COUNT__] = {__REGIME_THRESHOLDS__};
int RegimeModelIdx[__REGIME_COUNT__] = {__REGIME_MODEL_IDX__};
datetime LastModelLoad = 0;
int      DecisionLogHandle = INVALID_HANDLE;
int      UncertainLogHandle = INVALID_HANDLE;
int      DecisionSocket = INVALID_HANDLE;
int      NextDecisionId = 1;
int      ExecutedModelIdx = -1;
bool UseOnnxEncoder = false;
bool UseOnnxModel = false;
int      AdaptLogHandle = INVALID_HANDLE;
datetime LastAdaptRequest = 0;
string   LastTraceId = "";
string   LastSpanId  = "";
int      ShadowTradeHandle = INVALID_HANDLE;
bool     ShadowActive[];
bool     ShadowIsBuy[];
double   ShadowOpenPrice[];
double   ShadowSL[];
double   ShadowTP[];
double   ShadowLots[];
datetime ShadowOpenTime[];

string GenId(int bytes)
{
   string s = "";
   for(int i=0; i<bytes; i++)
   {
      int v = MathRand() & 0xFF;
      s += StringFormat("%02x", v);
   }
   return(s);
}

#import "onnxruntime_wrapper.ex4"
   int OnnxEncode(string model, double &inp[], double &out[]);
   int OnnxPredict(string model, double &inp[], double &out[]);
#import

//----------------------------------------------------------------------
// Model loading utilities
//----------------------------------------------------------------------

void UpdateKalman(double measurement)
{
   static bool initialized = false;
   double Q = 1e-5;
   double R = 1e-2;
   if(!initialized)
   {
      TrendEstimate = measurement;
      TrendVariance = 1.0;
      initialized = true;
   }
   TrendVariance += Q;
   double K = TrendVariance / (TrendVariance + R);
   TrendEstimate = TrendEstimate + K * (measurement - TrendEstimate);
   TrendVariance = (1 - K) * TrendVariance;
}

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

void ExtractJsonStringArray(string json, string key, string &arr[])
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
   {
      string s = StringTrimLeft(StringTrimRight(parts[i]));
      if(StringLen(s) >= 2 && StringMid(s,0,1)=="\"" && StringMid(s,StringLen(s)-1,1)=="\"")
         arr[i] = StringSubstr(s,1,StringLen(s)-2);
      else
         arr[i] = s;
   }
}

bool GetSymbolEmbedding(string sym, double &vec[])
{
   for(int i=0; i<SymbolEmbCount; i++)
   {
      if(SymbolEmbSymbols[i] == sym)
      {
         ArrayCopy(vec, SymbolEmbeddings[i]);
         return(true);
      }
   }
   return(false);
}

double GetRiskParityWeight(string sym)
{
   for(int i=0; i<ArraySize(RiskParitySymbols); i++)
   {
      if(RiskParitySymbols[i] == sym)
         return(RiskParityWeights[i]);
   }
   return(1.0);
}

bool ParseModelJson(string json)
{
   double tmp[];
   ExtractJsonArray(json, "\"coefficients\"", tmp);
   int n = MathMin(ArraySize(tmp), ArrayRange(ModelCoefficients,1));
   for(int i=0;i<n;i++)
      ModelCoefficients[0][i] = tmp[i];
   ExtractJsonArray(json, "\"hourly_thresholds\"", ModelThreshold);
   double prob_tmp[];
   ExtractJsonArray(json, "\"probability_table\"", prob_tmp);
   if(ArraySize(prob_tmp) == 24 && ArrayRange(ProbabilityLookup,0) > 0)
      for(int i=0;i<24;i++)
         ProbabilityLookup[0][i] = prob_tmp[i];
   ExtractJsonStringArray(json, "\"threshold_symbols\"", ThresholdSymbols);
   double th_tmp[];
   ExtractJsonArray(json, "\"threshold_table\"", th_tmp);
   int tsz = ArraySize(ThresholdSymbols);
   if(tsz > 0)
   {
      ArrayResize(ThresholdTable, tsz);
      int tcnt = MathMin(ArraySize(th_tmp), tsz*24);
      for(int i=0;i<tcnt;i++)
         ThresholdTable[i/24][i%24] = th_tmp[i];
   }
   ExtractJsonArray(json, "\"sl_coefficients\"", SLModelCoefficients);
   ExtractJsonArray(json, "\"tp_coefficients\"", TPModelCoefficients);
   ExtractJsonArray(json, "\"lot_coefficients\"", LotModelCoefficients);
   ExtractJsonStringArray(json, "\"risk_parity_symbols\"", RiskParitySymbols);
   ExtractJsonArray(json, "\"risk_parity_weights\"", RiskParityWeights);
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
   DefaultThreshold = ExtractJsonNumber(json, "\"threshold\"");
   ConformalLower = ExtractJsonNumber(json, "\"conformal_lower\"");
   ConformalUpper = ExtractJsonNumber(json, "\"conformal_upper\"");
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
   DefaultThreshold = StrToDouble(parts[1]);
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
   if(StringLen(ModelOnnxFile) > 0 && FileIsExist(ModelOnnxFile))
      UseOnnxModel = true;
   MarketBookAdd(SymbolToTrade);
   ShmRingInit("regime_ring", 1<<10);
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
               FileWrite(DecisionLogHandle, "event_id;timestamp;model_version;action;probability;sl_dist;tp_dist;model_idx;regime;chosen;risk_weight;variance;lots_predicted;executed_model_idx;features;trace_id;span_id");
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
         FileWrite(UncertainLogHandle,
                   "event_id;timestamp;model_version;action;probability;threshold;sl_dist;tp_dist;model_idx;regime;chosen;risk_weight;variance;lots_predicted;features;label");
      FileSeek(UncertainLogHandle, 0, SEEK_END);
   }
   else
      Print("Uncertain decision log open failed: ", GetLastError());
   if(StringLen(AdaptationLogFile) > 0)
   {
      AdaptLogHandle = FileOpen(AdaptationLogFile, FILE_CSV|FILE_WRITE|FILE_READ|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ, ';');
      if(AdaptLogHandle != INVALID_HANDLE)
      {
         if(FileSize(AdaptLogHandle) == 0)
            FileWrite(AdaptLogHandle, "timestamp;regime;old_weights;new_weights");
         FileSeek(AdaptLogHandle, 0, SEEK_END);
      }
      else
         Print("Adaptation log open failed: ", GetLastError());
   }
   if(EnableShadowTrading)
   {
      ShadowTradeHandle = FileOpen(ShadowTradesFile, FILE_CSV|FILE_WRITE|FILE_READ|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ, ';');
      if(ShadowTradeHandle != INVALID_HANDLE)
      {
         if(FileSize(ShadowTradeHandle) == 0)
            FileWrite(ShadowTradeHandle, "timestamp;model_idx;result;profit");
         FileSeek(ShadowTradeHandle, 0, SEEK_END);
      }
      else
         Print("Shadow trade log open failed: ", GetLastError());
      ArrayResize(ShadowActive, ModelCount);
      ArrayResize(ShadowIsBuy, ModelCount);
      ArrayResize(ShadowOpenPrice, ModelCount);
      ArrayResize(ShadowSL, ModelCount);
      ArrayResize(ShadowTP, ModelCount);
      ArrayResize(ShadowLots, ModelCount);
      ArrayResize(ShadowOpenTime, ModelCount);
      for(int i=0;i<ModelCount;i++)
         ShadowActive[i] = false;
   }
   ReplayDecisionLog(); // reprocess archived decisions when ReplayDecisions=true
   return(INIT_SUCCEEDED);
}

//----------------------------------------------------------------------
// Feature extraction utilities
//----------------------------------------------------------------------

double HourSin()
{
   double angle = 2.0 * M_PI * TimeHour(TimeCurrent()) / 24.0;
   return(MathSin(angle));
}

double HourCos()
{
   double angle = 2.0 * M_PI * TimeHour(TimeCurrent()) / 24.0;
   return(MathCos(angle));
}

double DowSin()
{
   double angle = 2.0 * M_PI * TimeDayOfWeek(TimeCurrent()) / 7.0;
   return(MathSin(angle));
}

double DowCos()
{
   double angle = 2.0 * M_PI * TimeDayOfWeek(TimeCurrent()) / 7.0;
   return(MathCos(angle));
}

double MonthSin()
{
   int month = TimeMonth(TimeCurrent());
   double angle = 2.0 * M_PI * (month - 1) / 12.0;
   return(MathSin(angle));
}

double MonthCos()
{
   int month = TimeMonth(TimeCurrent());
   double angle = 2.0 * M_PI * (month - 1) / 12.0;
   return(MathCos(angle));
}

double DomSin()
{
   int dom = TimeDay(TimeCurrent());
   double angle = 2.0 * M_PI * (dom - 1) / 31.0;
   return(MathSin(angle));
}

double DomCos()
{
   int dom = TimeDay(TimeCurrent());
   double angle = 2.0 * M_PI * (dom - 1) / 31.0;
   return(MathCos(angle));
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

double GetSlippage()
{
   return(LastSlippage);
}

double TradeDuration()
{
   for(int i = OrdersTotal() - 1; i >= 0; i--)
      if(OrderSelect(i, SELECT_BY_POS) &&
         OrderMagicNumber() == MagicNumber &&
         OrderSymbol() == SymbolToTrade)
         return(TimeCurrent() - OrderOpenTime());
   return(0.0);
}

double ExitReasonFlag(string reason)
{
   return(StringCompare(ExitReasonContext, reason) == 0 ? 1.0 : 0.0);
}

double CalendarImpactAt(datetime ts)
{
   double maxImp = 0.0;
   LastCalendarEventId = -1;
   for(int i=0; i<ArraySize(CalendarTimes); i++)
      if(MathAbs(ts - CalendarTimes[i]) <= EventWindowMinutes * 60)
         if(CalendarImpacts[i] > maxImp)
         {
            maxImp = CalendarImpacts[i];
            if(i < ArraySize(CalendarIds))
               LastCalendarEventId = CalendarIds[i];
         }
   return(maxImp);
}

double CalendarImpact()
{
   return(CalendarImpactAt(TimeCurrent()));
}

double CalendarFlag()
{
   return(CalendarImpact() > 0 ? 1.0 : 0.0);
}

int CalendarEventId()
{
   return(LastCalendarEventId);
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

void PollRegimeRing()
{
   uchar payload[4];
   int msg_type;
   int len = ArraySize(payload);
   while(ShmRingRead(msg_type, payload, len))
   {
      if(msg_type == MSG_REGIME && len > 0)
      {
         int reg = payload[0];
         if(reg != CurrentRegime)
         {
            CurrentRegime = reg;
            if(EnableDebugLogging)
               Print("Regime updated: ", reg);
         }
      }
      len = ArraySize(payload);
   }
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
      CachedBookSpread = ask - bid;
      CachedBidAskRatio = (ask>0) ? bid/ask : 0.0;
      BookImbalanceHist[BookImbPos] = CachedBookImbalance;
      BookImbPos = (BookImbPos + 1) % 5;
      if(BookImbCount < 5) BookImbCount++;
      double sum = 0.0;
      for(int j=0; j<BookImbCount; j++) sum += BookImbalanceHist[j];
      CachedBookImbalanceRoll = (BookImbCount>0) ? sum / BookImbCount : 0.0;
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

double BookSpread()
{
   RefreshBookCache();
   return(CachedBookSpread);
}

double BidAskRatio()
{
   RefreshBookCache();
   return(CachedBidAskRatio);
}

double BookImbalanceRoll()
{
   RefreshBookCache();
   return(CachedBookImbalanceRoll);
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

double CointegrationResidual(string sym1, string sym2="")
{
   if(sym2 == "")
   {
      sym2 = sym1;
      sym1 = SymbolToTrade;
   }
   double beta = 0.0;
   for(int i=0; i<ArraySize(CointBaseSymbols); i++)
   {
      if(CointBaseSymbols[i] == sym1 && CointPeerSymbols[i] == sym2)
      {
         beta = CointBetas[i];
         break;
      }
   }
   double p1 = iClose(sym1, 0, 0);
   double p2 = iClose(sym2, 0, 0);
   return(p1 - beta * p2);
}

double GraphDegree()
{
   for(int i=0; i<ArraySize(GraphSymbols); i++)
      if(GraphSymbols[i] == SymbolToTrade)
         return(GraphDegreeVals[i]);
   return(0.0);
}

double GraphPagerank()
{
   for(int i=0; i<ArraySize(GraphSymbols); i++)
      if(GraphSymbols[i] == SymbolToTrade)
         return(GraphPagerankVals[i]);
   return(0.0);
}

double GraphEmbedding(int idx)
{
   for(int i=0; i<ArraySize(GraphSymbols); i++)
      if(GraphSymbols[i] == SymbolToTrade)
         if(idx >= 0 && idx < GraphEmbDim)
            return(GraphEmbeddings[i][idx]);
   return(0.0);
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
   // Apply standardization using training set parameters when available
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

double ComputeEntryScore()
{
   double z = EntryIntercept;
   double feats[4];
   feats[0] = CachedBookImbalance;
   feats[1] = MarketInfo(SymbolToTrade, MODE_SPREAD);
   feats[2] = CachedBookBidVol;
   feats[3] = CachedBookAskVol;
   int n = MathMin(4, ArraySize(EntryCoefficients));
   for(int i=0; i<n; i++)
      z += EntryCoefficients[i] * feats[i];
   return(1.0 / (1.0 + MathExp(-z)));
}

double ComputeExitTime(double sl_dist, double tp_dist, double profit)
{
   double z = ExitIntercept;
   double feats[3];
   feats[0] = sl_dist;
   feats[1] = tp_dist;
   feats[2] = profit;
   int n = MathMin(3, ArraySize(ExitCoefficients));
   for(int i=0; i<n; i++)
      z += ExitCoefficients[i] * feats[i];
   if(z < 0) z = 0;
   return(z);
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

double ReplayProbability(double &feats[], int modelIdx)
{
   if(ModelHiddenSize > 0)
   {
      int hidden = ModelHiddenSize;
      int inputCount = ArraySize(NNLayer1Weights) / hidden;
      double z = NNLayer2Bias;
      for(int j=0; j<hidden; j++)
      {
         double h = NNLayer1Bias[j];
         for(int i=0; i<inputCount && i<ArraySize(feats); i++)
            h += NNLayer1Weights[j*inputCount + i] * feats[i];
         if(h < 0) h = 0;
         z += NNLayer2Weights[j] * h;
      }
      return(1.0 / (1.0 + MathExp(-z)));
   }
   double z = ModelIntercepts[modelIdx];
   int n = MathMin(FeatureCount, ArraySize(feats));
   for(int i=0; i<n; i++)
      z += ModelCoefficients[modelIdx][i] * feats[i];
   z = CalibrationCoef*z + CalibrationIntercept;
   return(1.0 / (1.0 + MathExp(-z)));
}

void ReplayDecisionLog()
{
   if(!ReplayDecisions)
      return;
   int h = FileOpen(DecisionLogFile, FILE_CSV|FILE_READ|FILE_TXT|FILE_SHARE_READ, ';');
   if(h == INVALID_HANDLE)
      return;
   if(!FileIsEnding(h))
   {
      string first = FileReadString(h);
      if(StringFind(first, "event_id") >= 0)
      {
         for(int i=0; i<9 && !FileIsEnding(h); i++)
            FileReadString(h);
      }
      else
         FileSeek(h, 0, SEEK_SET);
   }
   while(!FileIsEnding(h))
   {
      int event_id = (int)FileReadNumber(h);
      string ts = FileReadString(h);
      string mver = FileReadString(h);
      string action = FileReadString(h);
      double old_prob = FileReadNumber(h);
      double sl = FileReadNumber(h);
      double tp = FileReadNumber(h);
      int mIdx = (int)FileReadNumber(h);
      int reg = (int)FileReadNumber(h);
      string feat_str = FileReadString(h);
      string parts[];
      int cnt = StringSplit(feat_str, ',', parts);
      double feats[];
      ArrayResize(feats, cnt);
      for(int i=0; i<cnt; i++)
         feats[i] = StrToDouble(parts[i]);
      double new_prob = ReplayProbability(feats, mIdx);
      bool old_dec = old_prob >= DefaultThreshold;
      bool new_dec = new_prob >= DefaultThreshold;
      if(old_dec != new_dec)
         Print("Replay divergence event ", event_id,
               ": old=", DoubleToString(old_prob,3),
               " new=", DoubleToString(new_prob,3));
   }
   FileClose(h);
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
   if(UseOnnxModel)
   {
      double inp[];
      ArrayResize(inp, FeatureCount);
      for(int i=0; i<FeatureCount; i++)
         inp[i] = GetFeature(i);
      double out[];
      ArrayResize(out,1);
      if(OnnxPredict(ModelOnnxFile, inp, out) == 0)
         return(out[0]);
   }
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
   z *= GetRiskParityWeight(SymbolToTrade);
   if(z < MinLots) z = MinLots;
   if(z > MaxLots) z = MaxLots;
   return(z);
}

double GetTradeThreshold()
{
   if(CurrentRegime >= 0 && CurrentRegime < ArraySize(RegimeThresholds))
      return(RegimeThresholds[CurrentRegime]);
   int hr = TimeHour(TimeCurrent());
   for(int i=0;i<ArraySize(ThresholdSymbols);i++)
   {
      if(ThresholdSymbols[i] == SymbolToTrade && ArrayRange(ThresholdTable,0) > i)
         return(ThresholdTable[i][hr]);
   }
   if(ArraySize(ModelThreshold) == 24)
      return(ModelThreshold[hr]);
   return(DefaultThreshold);
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

void LogDecision(double &feats[], double prob, string action, int modelIdx, int regime, double riskWeight, double variance, int chosen)
{
   if(!EnableDecisionLogging)
      return;
   double sl_dist, tp_dist;
   CalcStops(sl_dist, tp_dist);
   double lots = CalcLots();
   string feat_vals = "";
   for(int i=0; i<FeatureCount; i++)
   {
      if(i>0) feat_vals += ",";
      feat_vals += DoubleToString(feats[i], 5);
   }
   datetime now = TimeCurrent();
   string trace_id = GenId(16);
   string span_id  = GenId(8);
   LastTraceId = trace_id;
   LastSpanId  = span_id;
   if(DecisionLogHandle != INVALID_HANDLE)
   {
      FileWrite(DecisionLogHandle, NextDecisionId, TimeToString(now, TIME_DATE|TIME_SECONDS), ModelVersion, action, prob, sl_dist, tp_dist, modelIdx, regime, chosen, riskWeight, variance, lots, ExecutedModelIdx, feat_vals, trace_id, span_id);
      FileFlush(DecisionLogHandle);
   }
   double thr = GetTradeThreshold();
   bool uncertain = MathAbs(prob - thr) <= UncertaintyMargin;
   if(uncertain && UncertainLogHandle != INVALID_HANDLE)
   {
      // capture feature snapshot for active learning
      FileWrite(UncertainLogHandle, NextDecisionId, TimeToString(now, TIME_DATE|TIME_SECONDS),
                ModelVersion, action, prob, thr, sl_dist, tp_dist, modelIdx, regime, chosen, riskWeight, variance, lots, feat_vals, "");
      FileFlush(UncertainLogHandle);
   }
   if(DecisionSocket != INVALID_HANDLE)
   {
      string json = StringFormat("{\"event_id\":%d,\"timestamp\":\"%s\",\"model_version\":\"%s\",\"action\":\"%s\",\"probability\":%.6f,\"sl_dist\":%.5f,\"tp_dist\":%.5f,\"model_idx\":%d,\"regime\":%d,\"chosen\":%d,\"risk_weight\":%.6f,\"variance\":%.6f,\"lots\":%.2f,\"executed_model_idx\":%d,\"features\":[%s],\"trace_id\":\"%s\",\"span_id\":\"%s\"}",
                                 NextDecisionId, TimeToString(now, TIME_DATE|TIME_SECONDS), ModelVersion, action, prob, sl_dist, tp_dist, modelIdx, regime, chosen, riskWeight, variance, lots, ExecutedModelIdx, feat_vals, trace_id, span_id);
      uchar bytes[];
      StringToCharArray(json+"\n", bytes, 0, WHOLE_ARRAY, CP_UTF8);
      SocketSend(DecisionSocket, bytes, ArraySize(bytes)-1);
   }
   NextDecisionId++;
}

int ExtractModelIndex(string comment)
{
   int pos = StringFind(comment, "model=");
   if(pos < 0) return(-1);
   pos += 6;
   int end = StringFind(comment, "|", pos);
   string val = (end > pos) ? StringSubstr(comment, pos, end-pos) : StringSubstr(comment, pos);
   return(StrToInteger(val));
}

int QueryBanditModel()
{
   string url = StringFormat("http://%s:%d/choose", BanditRouterHost, BanditRouterPort);
   uchar post[];
   uchar result[];
   string headers;
   int res = WebRequest("GET", url, "", 1000, post, result, headers);
   if(res != 200)
      return(-1);
   string s = CharArrayToString(result);
   return(StrToInteger(s));
}

void SendBanditReward(int modelIdx, double reward)
{
   string url = StringFormat("http://%s:%d/reward", BanditRouterHost, BanditRouterPort);
   string body = StringFormat("{\"model\":%d,\"reward\":%.2f}", modelIdx, reward);
   uchar post[];
   StringToCharArray(body, post, 0, WHOLE_ARRAY, CP_UTF8);
   uchar result[];
   string headers = "Content-Type: application/json\r\n";
   string resp_hdr;
   WebRequest("POST", url, headers, 1000, post, result, resp_hdr);
}

void LogAdaptationEvent(int regime, double &oldCoeffs[], double &newCoeffs[], double oldInt, double newInt)
{
   if(AdaptLogHandle == INVALID_HANDLE)
      return;
   string oldStr = "", newStr = "";
   for(int i=0;i<ArraySize(oldCoeffs);i++)
   {
      if(i>0){ oldStr += "|"; newStr += "|"; }
      oldStr += DoubleToString(oldCoeffs[i],8);
      newStr += DoubleToString(newCoeffs[i],8);
   }
   FileWrite(AdaptLogHandle, TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS), regime, oldStr+":"+DoubleToString(oldInt,8), newStr+":"+DoubleToString(newInt,8));
}

bool RequestAdaptedWeights(int regime)
{
   int sock = SocketCreate();
   if(sock == INVALID_HANDLE)
      return(false);
   if(!SocketConnect(sock, MetaAdaptHost, MetaAdaptPort, 1000))
   {
      SocketClose(sock);
      return(false);
   }
   string msg = StringFormat("{\"regime\":%d}\n", regime);
   uchar bytes[];
   StringToCharArray(msg, bytes, 0, WHOLE_ARRAY, CP_UTF8);
   if(SocketSend(sock, bytes, ArraySize(bytes)-1) <= 0)
   {
      SocketClose(sock);
      return(false);
   }
   uchar resp[2048];
   int got = SocketRead(sock, resp, 2047, 1000);
   SocketClose(sock);
   if(got <= 0)
      return(false);
   resp[got] = 0;
   string json = CharArrayToString(resp);
   double tmp[];
   ExtractJsonArray(json, "\"coefficients\"", tmp);
   int n = MathMin(ArraySize(tmp), ArrayRange(ModelCoefficients,1));
   double oldCoeffs[];
   ArrayResize(oldCoeffs, n);
   for(int i=0;i<n;i++)
   {
      oldCoeffs[i] = ModelCoefficients[0][i];
      ModelCoefficients[0][i] = tmp[i];
   }
   double oldInt = ModelIntercepts[0];
   double newInt = ExtractJsonNumber(json, "\"intercept\"");
   ModelIntercepts[0] = newInt;
   LogAdaptationEvent(regime, oldCoeffs, tmp, oldInt, newInt);
   return(true);
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
   PollRegimeRing();
   UpdateKalman(iClose(SymbolToTrade, 0, 0));
   // When ReloadModelInterval is set, periodically check for updated
   // coefficients written by ``online_trainer.py`` and reload them without
   // recompiling the EA.
   if(ReloadModelInterval > 0 && TimeCurrent() - LastModelLoad >= ReloadModelInterval)
   {
      if(LoadModel())
         Print("Model parameters reloaded");
      LastModelLoad = TimeCurrent();
   }
   if(AdaptationInterval > 0 && TimeCurrent() - LastAdaptRequest >= AdaptationInterval)
   {
      int reg = GetRegime();
      if(RequestAdaptedWeights(reg))
         Print("Adapted weights received");
      LastAdaptRequest = TimeCurrent();
   }
   if(EnableShadowTrading)
      ManageShadowTrades();
   else if(HasOpenOrders())
   {
      ManageOpenOrders();
      return;
   }

   UpdateFeatureHistory();
   int reg = QueryBanditModel();
   if(reg < 0)
      reg = SelectExpert();
   int modelIdx = reg;
   if(reg >= 0 && reg < ArraySize(RegimeModelIdx))
      modelIdx = RegimeModelIdx[reg];
   CurrentRegime = reg;
   ExecutedModelIdx = modelIdx;
   if(EnableDebugLogging)
      Print("Regime=", reg, " Model=", modelIdx);

   double feats[100];
   for(int i=0; i<FeatureCount && i<100; i++)
      feats[i] = GetFeature(i);

   double base_risk = AccountEquity() / AccountBalance();
   double probs[];
   ArrayResize(probs, ModelCount);
   double pvs[];
   ArrayResize(pvs, ModelCount);
   double risk_weights[];
   ArrayResize(risk_weights, ModelCount);
   for(int idx=0; idx<ModelCount; idx++)
   {
      pvs[idx] = ComputePredictiveVariance(idx);
      double rw = base_risk;
      if(pvs[idx] > 0.0)
         rw /= pvs[idx];
      double pr = ReplayProbability(feats, idx);
      LogDecision(feats, pr, "shadow", idx, reg, rw, pvs[idx], 0);
      probs[idx] = pr;
      risk_weights[idx] = rw;
   }

   double prob = probs[modelIdx];
   double pv = pvs[modelIdx];
   double risk_weight = risk_weights[modelIdx];

   if(prob < ConformalLower || prob > ConformalUpper)
      Print("Probability outside conformal bounds: " +
            DoubleToString(prob,4) + " not in [" +
            DoubleToString(ConformalLower,4) + "," +
            DoubleToString(ConformalUpper,4) + "]");

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
   if(!EnableShadowTrading && pv > MaxPredictiveVariance)
   {
      Print("Skipping trade due to high predictive variance: " + DoubleToString(pv, 6));
      int decision_id = NextDecisionId;
      LogDecision(feats, prob, "skip", modelIdx, reg, risk_weight, pv, 1);
      if(UncertainLogHandle != INVALID_HANDLE)
      {
         double sl_dist, tp_dist;
         CalcStops(sl_dist, tp_dist);
         string feat_vals = "";
         for(int i=0; i<FeatureCount; i++)
         {
            if(i>0) feat_vals += ",";
            feat_vals += DoubleToString(feats[i], 5);
         }
         datetime now = TimeCurrent();
         double thr = GetTradeThreshold();
         double lots = CalcLots();
         FileWrite(UncertainLogHandle, decision_id, TimeToString(now, TIME_DATE|TIME_SECONDS), ModelVersion, "skip", prob, thr, sl_dist, tp_dist, modelIdx, reg, 1, risk_weight, pv, lots, feat_vals, "");
         FileFlush(UncertainLogHandle);
      }
      return;
   }

   if(EnableShadowTrading)
   {
      for(int idx=0; idx<ModelCount; idx++)
      {
         if(ShadowActive[idx])
            continue;
         if(pvs[idx] > MaxPredictiveVariance)
            continue;
         double tradeLots = CalcLots();
         tradeLots *= risk_weights[idx];
         bool buy = (probs[idx] > EntryThreshold);
         ShadowActive[idx] = true;
         ShadowIsBuy[idx] = buy;
         ShadowOpenPrice[idx] = buy ? Ask : Bid;
         ShadowSL[idx] = buy ? GetNewSL(true) : GetNewSL(false);
         ShadowTP[idx] = buy ? GetNewTP(true) : GetNewTP(false);
         ShadowLots[idx] = tradeLots;
         ShadowOpenTime[idx] = TimeCurrent();
      }
      return;
   }

   // Open buy if probability exceeds threshold else sell
   double tradeLots = CalcLots();
   tradeLots *= risk_weight;
   int ticket;
   double thr = EntryThreshold;
   string action = (prob > thr) ? "buy" : "sell";
   int decision_id = NextDecisionId;
   LogDecision(feats, prob, action, modelIdx, reg, risk_weight, pv, 1);
   string order_comment = StringFormat("decision_id=%d;trace_id=%s;span_id=%s;model=%d", decision_id, LastTraceId, LastSpanId, modelIdx);
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
      double sl_dist = 0.0;
      double tp_dist = 0.0;
      if(OrderStopLoss() > 0)
         sl_dist = isBuy ? (price - OrderStopLoss())/Point : (OrderStopLoss() - price)/Point;
      if(OrderTakeProfit() > 0)
         tp_dist = isBuy ? (OrderTakeProfit() - price)/Point : (price - OrderTakeProfit())/Point;
      double cur_profit = OrderProfit() + OrderSwap() + OrderCommission();
      double exit_time = ComputeExitTime(sl_dist, tp_dist, cur_profit);
      double age = TimeCurrent() - OrderOpenTime();
      if(exit_time <= 0 || age >= exit_time)
      {
         if(!OrderClose(OrderTicket(), OrderLots(), price, 3))
            Print("OrderClose error: ", GetLastError());
         continue;
      }

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

void ManageShadowTrades()
{
   for(int idx=0; idx<ArraySize(ShadowActive); idx++)
   {
      if(!ShadowActive[idx])
         continue;
      bool isBuy = ShadowIsBuy[idx];
      double price = isBuy ? Bid : Ask;
      bool closed = false;
      string result = "";
      if(isBuy)
      {
         if(ShadowSL[idx] > 0 && price <= ShadowSL[idx])
         {
            closed = true;
            result = "sl";
         }
         else if(ShadowTP[idx] > 0 && price >= ShadowTP[idx])
         {
            closed = true;
            result = "tp";
         }
      }
      else
      {
         if(ShadowSL[idx] > 0 && price >= ShadowSL[idx])
         {
            closed = true;
            result = "sl";
         }
         else if(ShadowTP[idx] > 0 && price <= ShadowTP[idx])
         {
            closed = true;
            result = "tp";
         }
      }

      if(closed)
      {
         double profit = (isBuy ? (price - ShadowOpenPrice[idx]) : (ShadowOpenPrice[idx] - price)) * ShadowLots[idx];
         if(ShadowTradeHandle != INVALID_HANDLE)
         {
            datetime now = TimeCurrent();
            FileWrite(ShadowTradeHandle, TimeToString(now, TIME_DATE|TIME_SECONDS), idx, result, profit);
            FileFlush(ShadowTradeHandle);
         }
         ShadowActive[idx] = false;
      }
   }
}

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &req,
                        const MqlTradeResult  &res)
{
   if(trans.type!=TRADE_TRANSACTION_DEAL_ADD && trans.type!=TRADE_TRANSACTION_DEAL_UPDATE)
      return;
   if(!HistoryDealSelect(trans.deal))
      return;
   LastSlippage = trans.price - req.price;
   string comment = HistoryDealGetString(trans.deal, DEAL_COMMENT);
   int idx = ExtractModelIndex(comment);
   if(idx < 0) return;
   double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT)+
                   HistoryDealGetDouble(trans.deal, DEAL_SWAP)+
                   HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
   double reward = (profit > 0) ? 1.0 : 0.0;
   SendBanditReward(idx, reward);
}

void OnDeinit(const int reason)
{
   if(DecisionLogHandle != INVALID_HANDLE)
      FileClose(DecisionLogHandle);
   if(UncertainLogHandle != INVALID_HANDLE)
      FileClose(UncertainLogHandle);
   if(DecisionSocket != INVALID_HANDLE)
      SocketClose(DecisionSocket);
   if(AdaptLogHandle != INVALID_HANDLE)
      FileClose(AdaptLogHandle);
   if(ShadowTradeHandle != INVALID_HANDLE)
      FileClose(ShadowTradeHandle);
   MarketBookRelease(SymbolToTrade);
}
