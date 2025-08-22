#property strict
#include "model_interface.mqh"
#include <Arrays/ArrayInt.mqh>

#import "observer_capnp.dll"
int SerializeTradeEvent(int schema_version, int event_id, string trace_id, string event_time, string broker_time, string local_time, string action, int ticket, int magic, string source, string symbol, int order_type, double lots, double price, double sl, double tp, double profit, double profit_after_trade, double spread, string comment, double remaining_lots, double slippage, int volume, string open_time, double book_bid_vol, double book_ask_vol, double book_imbalance, double sl_hit_dist, double tp_hit_dist, double equity, double margin_level, double commission, double swap, int decision_id, string exit_reason, int duration_sec, uchar &out[]);
int SerializeMetrics(int schema_version, string time, int magic, double win_rate, double avg_profit, int trade_count, double drawdown, double sharpe, int file_write_errors, int socket_errors, double cpu_load, int book_refresh_seconds, int var_breach_count, int trade_queue_depth, int metric_queue_depth, int fallback_events, int fallback_logging, int wal_size, int trade_retry_count, int metric_retry_count, int anomaly_pending, int anomaly_late_count, uchar &out[]);
#import
#import "flight_client.dll"
bool FlightClientInit(string host, int port);
bool FlightClientSend(string path, uchar &payload[], int len);
#import

#import "kernel32.dll"
int GetEnvironmentVariableA(string name, uchar &buffer[], int size);
int WinExec(string cmd, int uCmdShow);
#import

string GetEnv(string name)
{
   uchar buffer[];
   ArrayResize(buffer, 4096);
   int len = GetEnvironmentVariableA(name, buffer, 4096);
   if(len<=0 || len>=4096)
      return "";
   return CharArrayToString(buffer, 0, len, CP_ACP);
}

string DefaultLogDir()
{
   string base = GetEnv("XDG_DATA_HOME");
   if(StringLen(base)==0)
   {
      string home = GetEnv("HOME");
      if(StringLen(home)==0)
         home = "";
      base = home + "/.local/share";
   }
   return base + "/botcopier/logs";
}

extern string TargetMagicNumbers = "12345,23456";
extern int    LearningExportIntervalMinutes = 15;
extern int    PredictionWindowSeconds       = 60;
extern double LotSizeTolerancePct           = 20.0;
extern double PriceTolerancePips            = 5.0;
extern bool   EnableLiveCloneMode           = false;
extern int    MaxModelsToRetain             = 3;
extern int    MetricsRollingDays            = 7;
extern int    MetricsDaysToKeep             = 30;
extern string LogDirectoryName              = DefaultLogDir(); // resume event_id from existing logs, start at 1 if none
extern bool   EnableDebugLogging            = false;
extern bool   UseBrokerTime                 = true;
extern string SymbolsToTrack                = ""; // empty=all
extern string CommitHash                   = "";
extern string ModelVersion                 = "";
extern string ModelFileName                = "model.json";
extern string TraceId                      = "";
extern int    BookRefreshSeconds           = 5;
extern string AnomalyServiceUrl            = "http://127.0.0.1:8000/anomaly";
extern double AnomalyThreshold             = 0.1;
extern int    AnomalyTimeoutSeconds       = 5;
extern string OtelEndpoint                = "";
extern string ModelStateFile              = "model_online.json";
extern string FlightServerHost            = "127.0.0.1";
extern int    FlightServerPort            = 8815;
extern int    FallbackRetryThreshold      = 5;
extern string CalendarFileName           = "calendar.csv";
extern int    CalendarEventWindowMinutes = 60;

int timer_handle;

int      tracked_tickets[];
CArrayInt ticket_map;
int      target_magics[];
string   track_symbols[];
datetime last_export = 0;
int      trade_log_handle = INVALID_HANDLE;
int      log_db_handle    = INVALID_HANDLE;
string   trade_log_buffer[];
int      NextEventId = 1;
datetime ModelTimestamp = 0;
int      FileWriteErrors = 0;
int      SocketErrors = 0;
int      FallbackEvents = 0;
int      TradeQueueDepth = 0;
int      MetricQueueDepth = 0;
string   log_dir = "";
const int SCHEMA_VERSION = 2;
int      AnomalyLateResponses = 0;

double   CpuLoad = 0.0;
int      CachedBookRefreshSeconds = 0;
string   RiskParitySymbols[];
double   RiskParityWeights[];
double   trend_estimate = 0.0;
double   trend_variance = 1.0;

datetime CalendarTimes[];
double   CalendarImpacts[];
int      CalendarIds[];

void LoadCalendar()
{
   ArrayResize(CalendarTimes, 0);
   ArrayResize(CalendarImpacts, 0);
   ArrayResize(CalendarIds, 0);
   int h = FileOpen(CalendarFileName, FILE_READ|FILE_CSV|FILE_ANSI);
   if(h==INVALID_HANDLE)
      return;
   // skip header if present
   // skip header row if present
   if(!FileIsEnding(h))
   {
      string hdr = FileReadString(h);
      if(StringCompare(hdr, "time")!=0)
         FileSeek(h, 0, SEEK_SET);
      else
      {
         FileReadString(h); // impact header
         FileReadString(h); // id header
      }
   }
   while(!FileIsEnding(h))
   {
      string ts = FileReadString(h);
      if(StringLen(ts)==0)
         break;
      double imp = FileReadNumber(h);
      int eid = (int)FileReadNumber(h);
      datetime t = StrToTime(ts);
      int idx = ArraySize(CalendarTimes);
      ArrayResize(CalendarTimes, idx+1);
      ArrayResize(CalendarImpacts, idx+1);
      ArrayResize(CalendarIds, idx+1);
      CalendarTimes[idx] = t;
      CalendarImpacts[idx] = imp;
      CalendarIds[idx] = eid;
   }
   FileClose(h);
}

int CalendarEventIdAt(datetime ts)
{
   double maxImp = 0.0;
   int best = -1;
   for(int i=0; i<ArraySize(CalendarTimes); i++)
      if(MathAbs(ts - CalendarTimes[i]) <= CalendarEventWindowMinutes * 60)
         if(CalendarImpacts[i] > maxImp)
         {
            maxImp = CalendarImpacts[i];
            best = CalendarIds[i];
         }
   return(best);
}

class PendingTrade
{
public:
   int      id;
   string   span_id;
   string   action;
   int      ticket;
   int      magic;
   string   source;
   string   symbol;
   int      order_type;
   double   lots;
   double   price;
   double   req_price;
   double   sl;
   double   tp;
   double   profit;
   double   profit_after;
   double   remaining;
   datetime time_event;
   string   comment;
   double   volume;
   datetime open_time;
   double   book_bid_vol;
   double   book_ask_vol;
   double   book_imbalance;
   double   spread;
   double   slippage;
   double   sl_hit_dist;
   double   tp_hit_dist;
   double   equity;
   double   margin_level;
   double   commission;
   double   swap;
   double   risk_weight;
   double   trend_estimate;
   double   trend_variance;
   int      decision_id;
   string   exit_reason;
   int      duration_sec;
   string   comment_with_span;
   string   open_time_str;
   datetime start_time;
   bool     anomaly_sent;
   string   anomaly_status;
   string   anomaly_id;
   datetime anomaly_sent_time;
};

PendingTrade AnomalyQueue[];
int AnomalyQueueDepth = 0;

uchar   pending_trades[][1];
string  pending_trade_lines[];
uchar   pending_metrics[][1];
string  pending_metric_lines[];
uchar   last_trade_payload[];
bool    have_last_trade = false;
uchar   last_metric_payload[];
bool    have_last_metric = false;
int     trade_backoff = 1;
int     metric_backoff = 1;
datetime next_trade_flush = 0;
datetime next_metric_flush = 0;
int     trade_retry_count = 0;
int     metric_retry_count = 0;

void EnqueueAnomaly(PendingTrade &t)
{
   t.anomaly_sent = false;
   t.anomaly_status = "";
   t.anomaly_id = "";
   t.anomaly_sent_time = 0;
   int n = ArraySize(AnomalyQueue);
   ArrayResize(AnomalyQueue, n+1);
   AnomalyQueue[n] = t;
    AnomalyQueueDepth = n+1;
}

void RemoveAnomaly(int index)
{
   int n = ArraySize(AnomalyQueue);
   for(int i=index; i<n-1; i++)
      AnomalyQueue[i] = AnomalyQueue[i+1];
   ArrayResize(AnomalyQueue, n-1);
   AnomalyQueueDepth = n-1;
}


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

void SendOtelSpan(string trace_id, string span_id, string name)
{
   if(StringLen(OtelEndpoint)==0)
      return;
   long now = (long)TimeCurrent();
   long ts = now * 1000000000;
   string payload = StringFormat(
      "{\"resourceSpans\":[{\"scopeSpans\":[{\"spans\":[{\"traceId\":\"%s\",\"spanId\":\"%s\",\"name\":\"%s\",\"startTimeUnixNano\":\"%I64d\",\"endTimeUnixNano\":\"%I64d\"}]}]}]}",
      trace_id, span_id, name, ts, ts);
   uchar data[]; StringToCharArray(payload, data);
   uchar result[]; string headers = "Content-Type: application/json"; string rheaders="";
   WebRequest("POST", OtelEndpoint, headers, 5000, data, ArraySize(data)-1, result, rheaders);
}

enum LogBackend
{
   LOG_BACKEND_SQLITE = 0,
   LOG_BACKEND_CSV    = 1
};

int CurrentBackend = LOG_BACKEND_CSV;

string   book_symbols[];
double   book_bid_cache[];
double   book_ask_cache[];
double   book_imb_cache[];
datetime book_last_refresh[];

int FindBookIndex(string symbol)
{
   for(int i=0; i<ArraySize(book_symbols); i++)
   {
      if(book_symbols[i] == symbol)
         return(i);
   }
   return(-1);
}

int MapGet(int key)
{
   return(ticket_map.Search(key));
}

void MapAdd(int key)
{
   if(ticket_map.Search(key) < 0)
      ticket_map.Add(key);
}

void MapRemove(int key)
{
   int pos = ticket_map.Search(key);
   if(pos >= 0)
      ticket_map.Delete(pos);
}

void UpdateKalman(double measurement)
{
   static bool initialized = false;
   double Q = 1e-5;
   double R = 1e-2;
   if(!initialized)
   {
      trend_estimate = measurement;
      trend_variance = 1.0;
      initialized = true;
   }
   trend_variance += Q;
   double K = trend_variance / (trend_variance + R);
   trend_estimate = trend_estimate + K * (measurement - trend_estimate);
   trend_variance = (1 - K) * trend_variance;
}

double GetCpuLoad()
{
   return(CpuLoad);
}

void UpdateCpuLoad(uint elapsed_ms)
{
   static double avg = 0.0;
   static int count = 0;
   if(count < 100)
      count++;
   avg = (avg*(count-1) + elapsed_ms) / count;
   CpuLoad = MathMin(avg / 200.0, 1.0);
   int interval = BookRefreshSeconds;
   if(CpuLoad < 0.25)
      interval = (int)MathMax(1.0, BookRefreshSeconds * 0.5);
   else if(CpuLoad > 0.75)
      interval = (int)(BookRefreshSeconds * 2.0);
   CachedBookRefreshSeconds = interval;
}


void EnqueueMessage(uchar &queue[][], uchar &msg[])
{
   int idx = ArraySize(queue);
   ArrayResize(queue, idx+1);
   ArrayResize(queue[idx], ArraySize(msg));
   ArrayCopy(queue[idx], msg);
}

void RemoveFirst(uchar &queue[][])
{
   int n = ArraySize(queue);
   if(n <= 1)
   {
      ArrayResize(queue, 0);
      return;
   }
   for(int i=0; i<n-1; i++)
      ArrayCopy(queue[i], queue[i+1]);
   ArrayResize(queue, n-1);
}

void RemoveFirstStr(string &queue[])
{
   int n = ArraySize(queue);
   if(n <= 1)
   {
      ArrayResize(queue, 0);
      return;
   }
   for(int i=0; i<n-1; i++)
      queue[i] = queue[i+1];
   ArrayResize(queue, n-1);
}

void EnqueuePending(uchar &queue[][], string &lines[], uchar &msg[], string line, string wal_fname)
{
   int idx = ArraySize(queue);
   ArrayResize(queue, idx+1);
   ArrayResize(queue[idx], ArraySize(msg));
   ArrayCopy(queue[idx], msg);
   ArrayResize(lines, idx+1);
   lines[idx] = line;
   AppendWal(wal_fname, msg);
}

bool LogToJournald(string tag, string line)
{
   string cmd = StringFormat("cmd.exe /c echo %s | systemd-cat -t %s", line, tag);
   int r = WinExec(cmd, 0);
   return(r > 31);
}

void FallbackLog(string tag, uchar &payload[], string line)
{
   if(StringLen(line)>0 && LogToJournald("botcopier-" + tag, line))
      return;

   string dir = StringReplace(LogDirectoryName, "\\", "/");
   string fname = dir + "/" + tag + "_fallback.log";
   int h = FileOpen(fname, FILE_READ|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ);
   if(h==INVALID_HANDLE)
      h = FileOpen(fname, FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE);
   if(h!=INVALID_HANDLE)
   {
      FileSeek(h, 0, SEEK_END);
      string out_line = line;
      if(StringLen(out_line)==0)
      {
         out_line = CharArrayToString(payload, 0, ArraySize(payload));
      }
      int _wr = FileWrite(h, out_line);
      if(_wr <= 0)
         FileWriteErrors++;
      FileClose(h);
   }
   else
   {
      FileWriteErrors++;
   }
}

void FlushPending(datetime now)
{
   if(ArraySize(pending_trades) > 0 && now >= next_trade_flush)
   {
      if(FlightClientSend("trades", pending_trades[0], ArraySize(pending_trades[0])))
      {
         RemoveFirst(pending_trades);
         RemoveFirstStr(pending_trade_lines);
         TradeQueueDepth = ArraySize(pending_trades);
         trade_backoff = 1;
         next_trade_flush = now;
         trade_retry_count = 0;
         SaveQueue(log_dir + "/pending_trades.wal", pending_trades);
      }
      else
      {
         SocketErrors++;
         trade_retry_count++;
         FallbackEvents++;
         trade_backoff = MathMin(trade_backoff*2, 3600);
         next_trade_flush = now + trade_backoff;
         if(trade_retry_count >= FallbackRetryThreshold)
         {
            string line = ArraySize(pending_trade_lines) > 0 ? pending_trade_lines[0] : "";
            FallbackLog("trades", pending_trades[0], line);
            trade_retry_count = 0;
         }
      }
   }

   if(ArraySize(pending_metrics) > 0 && now >= next_metric_flush)
   {
      if(FlightClientSend("metrics", pending_metrics[0], ArraySize(pending_metrics[0])))
      {
         RemoveFirst(pending_metrics);
         RemoveFirstStr(pending_metric_lines);
         MetricQueueDepth = ArraySize(pending_metrics);
         metric_backoff = 1;
         next_metric_flush = now;
         metric_retry_count = 0;
         SaveQueue(log_dir + "/pending_metrics.wal", pending_metrics);
      }
      else
      {
         SocketErrors++;
         metric_retry_count++;
         FallbackEvents++;
         metric_backoff = MathMin(metric_backoff*2, 3600);
         next_metric_flush = now + metric_backoff;
         if(metric_retry_count >= FallbackRetryThreshold)
         {
            string line = ArraySize(pending_metric_lines) > 0 ? pending_metric_lines[0] : "";
            FallbackLog("metrics", pending_metrics[0], line);
            metric_retry_count = 0;
         }
      }
   }
}

void AppendWal(string fname, uchar &msg[])
{
   int h = FileOpen(fname, FILE_BIN|FILE_WRITE|FILE_READ|FILE_COMMON);
   if(h==INVALID_HANDLE)
      h = FileOpen(fname, FILE_BIN|FILE_WRITE|FILE_COMMON);
   if(h==INVALID_HANDLE)
      return;
   FileSeek(h, 0, SEEK_END);
   int len = ArraySize(msg);
   FileWriteInteger(h, len);
   FileWriteArray(h, msg, 0, len);
   FileClose(h);
}

void SaveQueue(string fname, uchar &queue[][])
{
   int h = FileOpen(fname, FILE_BIN|FILE_WRITE|FILE_COMMON);
   if(h==INVALID_HANDLE)
      return;
   for(int i=0; i<ArraySize(queue); i++)
   {
      int len = ArraySize(queue[i]);
      FileWriteInteger(h, len);
      FileWriteArray(h, queue[i], 0, len);
   }
   FileClose(h);
}

void LoadQueue(string fname, uchar &queue[][])
{
   int h = FileOpen(fname, FILE_BIN|FILE_READ|FILE_COMMON);
   if(h==INVALID_HANDLE)
      return;
   while(!FileIsEnding(h))
   {
      int len = FileReadInteger(h);
      if(len <= 0)
         break;
      int idx = ArraySize(queue);
      ArrayResize(queue, idx+1);
      ArrayResize(queue[idx], len);
      FileReadArray(h, queue[idx], 0, len);
   }
   FileClose(h);
   FileDelete(fname);
}

void ReplayWal(string fname, uchar &queue[][])
{
   int h = FileOpen(fname, FILE_BIN|FILE_READ|FILE_COMMON);
   if(h==INVALID_HANDLE)
      return;
   while(!FileIsEnding(h))
   {
      int len = FileReadInteger(h);
      if(len <= 0)
         break;
      int idx = ArraySize(queue);
      ArrayResize(queue, idx+1);
      ArrayResize(queue[idx], len);
      FileReadArray(h, queue[idx], 0, len);
   }
   FileClose(h);
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

void LoadModelState()
{
   int h = FileOpen(ModelStateFile, FILE_READ|FILE_SHARE_READ|FILE_COMMON);
   if(h==INVALID_HANDLE)
      return;
   string content = "";
   while(!FileIsEnding(h))
      content += FileReadString(h);
   FileClose(h);
   int last_id = (int)ExtractJsonNumber(content, "\"last_event_id\"");
   if(last_id > 0)
      NextEventId = last_id + 1;
}

void SaveModelState()
{
   string tmp = ModelStateFile + ".tmp";
   int h = FileOpen(tmp, FILE_WRITE|FILE_TXT|FILE_COMMON);
   if(h==INVALID_HANDLE)
      return;
   string json = StringFormat("{\"last_event_id\":%d}", NextEventId - 1);
   int _wr = FileWrite(h, json);
   if(_wr <= 0)
      FileWriteErrors++;
   FileClose(h);
   FileDelete(ModelStateFile);
   FileMove(tmp, ModelStateFile, FILE_COMMON);
}

int GetMaxEventIdFromLogs(string dir)
{
   int max_id = 0;
   string fname = dir + "/trades_raw.csv";
   if(FileIsExist(fname))
   {
      int h = FileOpen(fname, FILE_CSV|FILE_READ|FILE_TXT|FILE_SHARE_READ|FILE_SHARE_WRITE, ';');
      if(h!=INVALID_HANDLE)
      {
         string last_line = "";
         while(!FileIsEnding(h))
         {
            string line = FileReadString(h);
            if(StringLen(line) > 0)
               last_line = line;
         }
         FileClose(h);
         if(StringLen(last_line) > 0)
         {
            string parts[];
            if(StringSplit(last_line, ';', parts) > 0)
            {
               int id = (int)StringToInteger(parts[0]);
               if(id > max_id)
                  max_id = id;
            }
         }
      }
   }
   string file = "";
   int handle = FileFindFirst(dir + "/*.*", file);
   if(handle!=INVALID_HANDLE)
   {
      do
      {
         if(StringFind(file, "trades_raw.csv") >= 0)
            continue;
         int pos = -1;
         if(StringSubstr(file, StringLen(file)-4, 4) == ".csv")
            pos = StringLen(file)-4;
         else if(StringLen(file) >= 7 && StringSubstr(file, StringLen(file)-7, 7) == ".csv.gz")
            pos = StringLen(file)-7;
         if(pos >= 0)
         {
            string name = StringSubstr(file, 0, pos);
            string parts2[];
            int cnt = StringSplit(name, '_', parts2);
            if(cnt > 0)
            {
               int id2 = (int)StringToInteger(parts2[cnt-1]);
               if(id2 > max_id)
                  max_id = id2;
            }
         }
      }
      while(FileFindNext(handle, file));
      FileFindClose(handle);
   }
   return(max_id);
}

void LoadRiskWeights()
{
   int h = FileOpen(ModelFileName, FILE_READ|FILE_SHARE_READ|FILE_COMMON);
   if(h==INVALID_HANDLE)
      return;
   string content = "";
   while(!FileIsEnding(h))
      content += FileReadString(h);
   FileClose(h);
   ExtractJsonStringArray(content, "\"risk_parity_symbols\"", RiskParitySymbols);
   ExtractJsonArray(content, "\"risk_parity_weights\"", RiskParityWeights);
}

double GetRiskWeight(string sym)
{
   for(int i=0; i<ArraySize(RiskParitySymbols); i++)
   {
       if(RiskParitySymbols[i] == sym)
           return(RiskParityWeights[i]);
   }
   return(1.0);
}

bool CheckAnomaly(string job_id, double price, double sl, double tp, double lots, double spread, double slippage)
{
   string payload = StringFormat("{\"id\":\"%s\",\"payload\":[%.5f,%.5f,%.5f,%.2f,%.5f,%.5f]}", job_id, price, sl, tp, lots, spread, slippage);
   uchar data[];
   StringToCharArray(payload, data);
   uchar result[];
   string headers = "Content-Type: application/json";
   string rheaders = "";
   int res = WebRequest("POST", AnomalyServiceUrl, headers, 1000, data, ArraySize(data)-1, result, rheaders);
   return(res>=200 && res<300);
}

int PollAnomaly(string job_id, double &err)
{
   string url = AnomalyServiceUrl + "?id=" + job_id;
   uchar data[];
   uchar result[];
   string rheaders = "";
   int res = WebRequest("GET", url, "", 1000, data, 0, result, rheaders);
   if(res==200)
   {
      string txt = CharArrayToString(result);
      err = StrToDouble(txt);
      return(1);
   }
   if(res==404)
      return(0);
   return(-1);
}

bool Contains(int &arr[], int value)
{
   int left = 0;
   int right = ArraySize(arr) - 1;
   while(left <= right)
   {
      int mid = (left + right) / 2;
      int v = arr[mid];
      if(v == value)
         return(true);
      if(v < value)
         left = mid + 1;
      else
         right = mid - 1;
   }
   return(false);
}

void GetBookVolumes(string symbol, double &bid_vol, double &ask_vol, double &imbalance)
{
   int idx = FindBookIndex(symbol);
   if(idx < 0)
   {
      int sz = ArraySize(book_symbols);
      ArrayResize(book_symbols, sz+1);
      ArrayResize(book_bid_cache, sz+1);
      ArrayResize(book_ask_cache, sz+1);
      ArrayResize(book_imb_cache, sz+1);
      ArrayResize(book_last_refresh, sz+1);
      idx = sz;
      book_symbols[idx] = symbol;
      book_bid_cache[idx] = 0.0;
      book_ask_cache[idx] = 0.0;
      book_imb_cache[idx] = 0.0;
      book_last_refresh[idx] = 0;
   }

   bid_vol = book_bid_cache[idx];
   ask_vol = book_ask_cache[idx];
   imbalance = book_imb_cache[idx];

   datetime now = UseBrokerTime ? TimeCurrent() : TimeLocal();
   if(now - book_last_refresh[idx] < CachedBookRefreshSeconds)
      return;

   MqlBookInfo book[];
   double b = 0.0;
   double a = 0.0;
   if(MarketBookGet(symbol, book))
   {
      for(int i=0; i<ArraySize(book); i++)
      {
         if(book[i].type==BOOK_TYPE_BUY)
            b += book[i].volume;
         else if(book[i].type==BOOK_TYPE_SELL)
            a += book[i].volume;
      }
      double imb = 0.0;
      if(b + a > 0)
         imb = (b - a) / (b + a);
      bid_vol = b;
      ask_vol = a;
      imbalance = imb;
      book_bid_cache[idx] = b;
      book_ask_cache[idx] = a;
      book_imb_cache[idx] = imb;
      book_last_refresh[idx] = now;
   }
}

int OnInit()
{
   EventSetTimer(1);
   MathSrand(GetTickCount());
   LoadCalendar();
   if(StringLen(TraceId)==0)
      TraceId = GenId(16);
   ArrayResize(tracked_tickets, 0);
   ticket_map.Clear();

   string parts[];
   int cnt = StringSplit(TargetMagicNumbers, ',', parts);
   ArrayResize(target_magics, cnt);
   for(int i=0; i<cnt; i++)
      target_magics[i] = (int)StringToInteger(StringTrimLeft(StringTrimRight(parts[i])));

   int sym_cnt = StringSplit(SymbolsToTrack, ',', parts);
   if(sym_cnt==1 && StringLen(parts[0])==0)
      sym_cnt = 0;
   ArrayResize(track_symbols, sym_cnt);
   for(int j=0; j<sym_cnt; j++)
      track_symbols[j] = StringTrimLeft(StringTrimRight(parts[j]));

   log_dir = StringReplace(LogDirectoryName, "\\", "/");
   if(!FileIsExist(log_dir))
   {
      if(FolderCreate(log_dir))
         Print("Created log directory: " + log_dir);
      else
         Print("Failed to create log directory: " + log_dir);
   }
   else
      Print("Log directory exists: " + log_dir);
   LoadQueue(log_dir + "/pending_trades.bin", pending_trades);
   LoadQueue(log_dir + "/pending_metrics.bin", pending_metrics);
   ReplayWal(log_dir + "/pending_trades.wal", pending_trades);
   ReplayWal(log_dir + "/pending_metrics.wal", pending_metrics);
   TradeQueueDepth = ArraySize(pending_trades);
   MetricQueueDepth = ArraySize(pending_metrics);
   FlightClientInit(FlightServerHost, FlightServerPort);
   LoadModelState();
   int max_id = GetMaxEventIdFromLogs(log_dir);
   if(max_id >= NextEventId)
      NextEventId = max_id + 1;
   LoadRiskWeights();
   ModelTimestamp = FileGetInteger(ModelStateFile, FILE_MODIFY_DATE);
   string symbols_json = "[";
   for(int k=0; k<ArraySize(track_symbols); k++)
   {
      if(k>0)
         symbols_json += ",";
      symbols_json += "\"" + track_symbols[k] + "\"";
   }
   symbols_json += "]";
   string run_info = "{";
   run_info += "\"commit_hash\":\"" + CommitHash + "\",";
   run_info += "\"model_version\":\"" + ModelVersion + "\",";
   run_info += "\"broker\":\"" + AccountInfoString(ACCOUNT_COMPANY) + "\",";
   run_info += "\"symbols\":" + symbols_json + ",";
   run_info += "\"mt4_build\":" + IntegerToString((int)TerminalInfoInteger(TERMINAL_BUILD));
   run_info += "}";
   int info_handle = FileOpen(log_dir + "/run_info.json", FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ);
   if(info_handle!=INVALID_HANDLE)
   {
      FileWriteString(info_handle, run_info);
      FileClose(info_handle);
   }

   CachedBookRefreshSeconds = BookRefreshSeconds;

   // Prefer SQLite backend, fall back to CSV
   string db_fname = log_dir + "/trades_raw.sqlite";
   log_db_handle = DatabaseOpen(db_fname, DATABASE_OPEN_READWRITE|DATABASE_OPEN_CREATE);
   if(log_db_handle!=INVALID_HANDLE)
   {
      string create_sql = "CREATE TABLE IF NOT EXISTS logs (event_id INTEGER, event_time TEXT, broker_time TEXT, local_time TEXT, action TEXT, ticket INTEGER, magic INTEGER, source TEXT, symbol TEXT, order_type INTEGER, lots REAL, price REAL, sl REAL, tp REAL, profit REAL, profit_after_trade REAL, spread INTEGER, trace_id TEXT, span_id TEXT, comment TEXT, remaining_lots REAL, slippage REAL, volume INTEGER, open_time TEXT, book_bid_vol REAL, book_ask_vol REAL, book_imbalance REAL, sl_hit_dist REAL, tp_hit_dist REAL, decision_id INTEGER, is_anomaly INTEGER, equity REAL, margin_level REAL, commission REAL, swap REAL, exit_reason TEXT, duration_sec INTEGER, calendar_event_id INTEGER)";
      DatabaseExecute(log_db_handle, create_sql);
      // Resume event id from existing records
      int stmt = DatabasePrepare(log_db_handle, "SELECT MAX(event_id) FROM logs");
      if(stmt!=INVALID_HANDLE)
      {
         if(DatabaseRead(stmt))
         {
            int last_id = (int)DatabaseGetInteger(stmt, 0);
            if(last_id > 0)
               NextEventId = last_id + 1;
         }
         DatabaseFinalize(stmt);
      }
      CurrentBackend = LOG_BACKEND_SQLITE;
      Print("Using SQLite log backend");
   }
   else
   {
      // Final fallback to CSV
      string log_fname = log_dir + "/trades_raw.csv";
      trade_log_handle = FileOpen(log_fname, FILE_CSV|FILE_WRITE|FILE_READ|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ, ';');
      if(trade_log_handle!=INVALID_HANDLE)
      {
         bool need_header = (FileSize(trade_log_handle)==0);
         int last_id = 0;
         if(!need_header)
         {
            FileSeek(trade_log_handle, 0, SEEK_SET);
            while(!FileIsEnding(trade_log_handle))
            {
               string field = FileReadString(trade_log_handle);
               if(field=="event_id")
               {
                  while(!FileIsLineEnding(trade_log_handle) && !FileIsEnding(trade_log_handle))
                     FileReadString(trade_log_handle);
                  continue;
               }
               int id = (int)StringToInteger(field);
               if(id > last_id)
                  last_id = id;
               while(!FileIsLineEnding(trade_log_handle) && !FileIsEnding(trade_log_handle))
                  FileReadString(trade_log_handle);
            }
            FileSeek(trade_log_handle, 0, SEEK_END);
         }
         if(last_id > 0)
            NextEventId = last_id + 1;
         if(need_header)
         {
         string header = "event_id;event_time;broker_time;local_time;action;ticket;magic;source;symbol;order_type;lots;price;sl;tp;profit;profit_after_trade;spread;trace_id;span_id;comment;remaining_lots;slippage;volume;open_time;book_bid_vol;book_ask_vol;book_imbalance;sl_hit_dist;tp_hit_dist;decision_id;is_anomaly;equity;margin_level;commission;swap;risk_weight;trend_estimate;trend_variance;exit_reason;duration_sec;calendar_event_id";
            int _wr = FileWrite(trade_log_handle, header);
            if(_wr <= 0)
               FileWriteErrors++;
         }
      }
      CurrentBackend = LOG_BACKEND_CSV;
      Print("Using CSV log backend");
   }

   last_export = UseBrokerTime ? TimeCurrent() : TimeLocal();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   FlushTradeBuffer();
   FlushMetricBuffer();
   SaveQueue(log_dir + "/pending_trades.bin", pending_trades);
   SaveQueue(log_dir + "/pending_metrics.bin", pending_metrics);
   SaveQueue(log_dir + "/pending_trades.wal", pending_trades);
   SaveQueue(log_dir + "/pending_metrics.wal", pending_metrics);
   if(trade_log_handle!=INVALID_HANDLE)
   {
      FileClose(trade_log_handle);
      trade_log_handle = INVALID_HANDLE;
   }
   if(log_db_handle!=INVALID_HANDLE)
   {
      DatabaseClose(log_db_handle);
      log_db_handle = INVALID_HANDLE;
   }
}

bool MagicMatches(int magic)
{
   for(int i=0; i<ArraySize(target_magics); i++)
      if(target_magics[i]==magic)
         return(true);
   return(false);
}

bool SymbolMatches(string symbol)
{
   if(ArraySize(track_symbols)==0)
      return(true);
   for(int i=0; i<ArraySize(track_symbols); i++)
      if(StringCompare(track_symbols[i], symbol, true)==0)
         return(true);
   return(false);
}

bool IsTracked(int ticket)
{
   return(MapGet(ticket)>=0);
}

void AddTicket(int ticket)
{
   int n = ArraySize(tracked_tickets);
   ArrayResize(tracked_tickets, n+1);
   tracked_tickets[n] = ticket;
   MapAdd(ticket);
}

void RemoveTicket(int ticket)
{
   int idx = -1;
   for(int i=0; i<ArraySize(tracked_tickets); i++)
   {
      if(tracked_tickets[i]==ticket)
      {
         idx = i;
         break;
      }
   }
   if(idx<0)
      return;
   for(int j=idx; j<ArraySize(tracked_tickets)-1; j++)
      tracked_tickets[j] = tracked_tickets[j+1];
   ArrayResize(tracked_tickets, ArraySize(tracked_tickets)-1);
   MapRemove(ticket);
}

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &req,
                        const MqlTradeResult  &res)
{
   datetime now = UseBrokerTime ? TimeCurrent() : TimeLocal();

   // Only process executed deals or order updates
   if(trans.type!=TRADE_TRANSACTION_DEAL_ADD &&
      trans.type!=TRADE_TRANSACTION_DEAL_UPDATE &&
      trans.type!=TRADE_TRANSACTION_ORDER_UPDATE)
      return;

   if(!HistoryDealSelect(trans.deal))
      return;

   int    entry      = (int)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   int    magic      = (int)HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
   string symbol     = HistoryDealGetString(trans.deal, DEAL_SYMBOL);
   int    order_type = (int)HistoryDealGetInteger(trans.deal, DEAL_TYPE);
   double lots       = HistoryDealGetDouble(trans.deal, DEAL_VOLUME);
   double price      = HistoryDealGetDouble(trans.deal, DEAL_PRICE);
   double sl         = HistoryDealGetDouble(trans.deal, DEAL_SL);
   double tp         = HistoryDealGetDouble(trans.deal, DEAL_TP);
   double profit     = HistoryDealGetDouble(trans.deal, DEAL_PROFIT)+
                       HistoryDealGetDouble(trans.deal, DEAL_SWAP)+
                       HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
   string comment    = HistoryDealGetString(trans.deal, DEAL_COMMENT);
   int    ticket     = (int)trans.order;


   if(!MagicMatches(magic) || !SymbolMatches(symbol))
      return;

   double remaining = 0.0;
   if(OrderSelect(ticket, SELECT_BY_TICKET, MODE_TRADES))
      remaining = OrderLots();

   double profit_after = AccountBalance() + AccountProfit();
   double slippage = trans.price - req.price;
   double equity   = AccountEquity();
   double margin   = AccountMarginLevel();
   double risk_weight = GetRiskWeight(symbol);
   lots *= risk_weight;
   remaining *= risk_weight;

   if(entry==DEAL_ENTRY_IN || entry==DEAL_ENTRY_INOUT)
   {
      double bid_vol, ask_vol, book_imb;
      GetBookVolumes(symbol, bid_vol, ask_vol, book_imb);
      int event_id = NextEventId++;
      LogTrade(event_id, "OPEN", ticket, magic, "mt4", symbol, order_type,
               lots, price, req.price, sl, tp, 0.0, profit_after,
               remaining, now, comment, iVolume(symbol, 0, 0), 0,
               bid_vol, ask_vol, book_imb, slippage, equity, margin, risk_weight);
      if(!IsTracked(ticket))
         AddTicket(ticket);
      else if(entry==DEAL_ENTRY_INOUT && remaining>0.0 && OrderSelect(ticket, SELECT_BY_TICKET, MODE_TRADES))
      {
         double cur_price = OrderOpenPrice();
         double cur_sl    = OrderStopLoss();
         double cur_tp    = OrderTakeProfit();
         double bid_vol2, ask_vol2, book_imb2;
         GetBookVolumes(symbol, bid_vol2, ask_vol2, book_imb2);
         event_id = NextEventId++;
         LogTrade(event_id, "MODIFY", ticket, magic, "mt4", symbol, order_type,
                  0.0, cur_price, cur_price, cur_sl, cur_tp, 0.0, profit_after,
                  remaining, now, comment, iVolume(symbol, 0, 0), 0,
                  bid_vol2, ask_vol2, book_imb2, slippage, equity, margin, risk_weight);
      }
   }
   else if(entry==DEAL_ENTRY_OUT || entry==DEAL_ENTRY_OUT_BY)
   {
      datetime open_time = 0;
      if(OrderSelect(ticket, SELECT_BY_TICKET, MODE_HISTORY))
         open_time = OrderOpenTime();
      else if(OrderSelect(ticket, SELECT_BY_TICKET, MODE_TRADES))
         open_time = OrderOpenTime();
      double bid_vol3, ask_vol3, book_imb3;
      GetBookVolumes(symbol, bid_vol3, ask_vol3, book_imb3);
      int event_id = NextEventId++;
      LogTrade(event_id, "CLOSE", ticket, magic, "mt4", symbol, order_type,
               lots, price, req.price, sl, tp, profit, profit_after,
               remaining, now, comment, iVolume(symbol, 0, 0), open_time,
               bid_vol3, ask_vol3, book_imb3, slippage, equity, margin, risk_weight);
      if(IsTracked(ticket) && remaining==0.0)
         RemoveTicket(ticket);
      else if(remaining>0.0 && OrderSelect(ticket, SELECT_BY_TICKET, MODE_TRADES))
      {
         double cur_price = OrderOpenPrice();
         double cur_sl    = OrderStopLoss();
         double cur_tp    = OrderTakeProfit();
         double bid_vol4, ask_vol4, book_imb4;
         GetBookVolumes(symbol, bid_vol4, ask_vol4, book_imb4);
         event_id = NextEventId++;
         LogTrade(event_id, "MODIFY", ticket, magic, "mt4", symbol, order_type,
                  0.0, cur_price, cur_price, cur_sl, cur_tp, 0.0, profit_after,
                  remaining, now, comment, iVolume(symbol, 0, 0), 0,
                  bid_vol4, ask_vol4, book_imb4, slippage, equity, margin, risk_weight);
      }
   }

   FlushTradeBuffer();
}

void OnTick()
{
   uint tick_start = GetTickCount();
   datetime now = UseBrokerTime ? TimeCurrent() : TimeLocal();
   UpdateKalman(iClose(Symbol(), 0, 0));
   datetime ts = FileGetInteger(ModelStateFile, FILE_MODIFY_DATE);
   if(ts != ModelTimestamp)
   {
      LoadModelState();
      ModelTimestamp = ts;
   }
   int current[];
   int cur_idx = 0;
   ArrayResize(current, OrdersTotal());

   for(int i=0; i<OrdersTotal(); i++)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
         continue;
      if(!MagicMatches(OrderMagicNumber()))
         continue;
      if(!SymbolMatches(OrderSymbol()))
         continue;

      int ticket = OrderTicket();
      current[cur_idx++] = ticket;

      if(!IsTracked(ticket))
      {
         double profit_after = AccountBalance() + AccountProfit();
         double bid_vol, ask_vol, book_imb;
         GetBookVolumes(OrderSymbol(), bid_vol, ask_vol, book_imb);
         int event_id = NextEventId++;
         double equity = AccountEquity();
         double margin = AccountMarginLevel();
         double risk_weight = GetRiskWeight(OrderSymbol());
         double lots = OrderLots() * risk_weight;
         LogTrade(event_id, "OPEN", ticket, OrderMagicNumber(), "mt4", OrderSymbol(), OrderType(),
                  lots, OrderOpenPrice(), OrderOpenPrice(), OrderStopLoss(), OrderTakeProfit(),
                  0.0, profit_after, lots, now, OrderComment(),
                  iVolume(OrderSymbol(), 0, 0), 0, bid_vol, ask_vol, book_imb,
                  0.0, equity, margin, risk_weight);
         AddTicket(ticket);
      }
   }
   ArrayResize(current, cur_idx);
   ArraySort(current, WHOLE_ARRAY, 0, MODE_ASCEND);

   for(int t=0; t<ArraySize(tracked_tickets); t++)
   {
      int ticket = tracked_tickets[t];
      bool still_open = Contains(current, ticket);
      if(!still_open)
      {
         if(OrderSelect(ticket, SELECT_BY_TICKET, MODE_HISTORY))
         {
             double profit_after2 = AccountBalance() + AccountProfit();
             double bid_vol, ask_vol, book_imb;
             GetBookVolumes(OrderSymbol(), bid_vol, ask_vol, book_imb);
             int event_id = NextEventId++;
             double equity = AccountEquity();
             double margin = AccountMarginLevel();
             double risk_weight = GetRiskWeight(OrderSymbol());
             double lots = OrderLots() * risk_weight;
             LogTrade(event_id, "CLOSE", ticket, OrderMagicNumber(), "mt4", OrderSymbol(),
                       OrderType(), lots, OrderClosePrice(), OrderClosePrice(), OrderStopLoss(),
                       OrderTakeProfit(), OrderProfit()+OrderSwap()+OrderCommission(),
                       profit_after2, 0.0, now, OrderComment(),
                       iVolume(OrderSymbol(), 0, 0), OrderOpenTime(),
                       bid_vol, ask_vol, book_imb, 0.0, equity, margin, risk_weight);
         }
         RemoveTicket(ticket);
         t--; // adjust index after removal
      }
   }
   uint elapsed = GetTickCount() - tick_start;
   UpdateCpuLoad(elapsed);
}

void OnTimer()
{
   ProcessAnomalyQueue();
   FlushTradeBuffer();
   FlushMetricBuffer();
   datetime now = UseBrokerTime ? TimeCurrent() : TimeLocal();
   FlushPending(now);

   if(now - last_export < LearningExportIntervalMinutes*60)
      return;

   ExportLogs(now);
   WriteMetrics(now);
   ManageMetrics(now);
   last_export = now;
}

string EscapeJson(string s)
{
   s = StringReplace(s, "\\", "\\\\");
   s = StringReplace(s, "\"", "\\\"");
   s = StringReplace(s, "\n", "\\n");
   s = StringReplace(s, "\r", "\\r");
   return(s);
}

bool SendTrade(uchar &payload[])
{
   int len = ArraySize(payload);
   uchar out[];
   bool send_full = true;
   if(have_last_trade && ArraySize(last_trade_payload)==len)
   {
      int changes = 0;
      for(int i=0; i<len; i++)
         if(payload[i] != last_trade_payload[i])
            changes++;
      if(changes>0 && changes<256)
      {
         send_full = false;
         ArrayResize(out, 3 + changes*3);
         out[0] = (uchar)SCHEMA_VERSION;
         out[1] = 1; // delta tag
         out[2] = (uchar)changes;
         int pos = 3;
         for(int i=0; i<len; i++)
         {
            if(payload[i] != last_trade_payload[i])
            {
               out[pos]   = (uchar)(i>>8);
               out[pos+1] = (uchar)(i & 0xFF);
               out[pos+2] = payload[i];
               pos += 3;
            }
         }
      }
   }
   if(send_full)
   {
      ArrayResize(out, len + 2);
      out[0] = (uchar)SCHEMA_VERSION;
      out[1] = 0; // full tag
      ArrayCopy(out, payload, 2, 0, len);
   }
   ArrayResize(last_trade_payload, len);
   ArrayCopy(last_trade_payload, payload, 0, 0, len);
   have_last_trade = true;
   uchar zipped[];
   if(!CryptEncode(CRYPT_ARCHIVE_GZIP, out, zipped))
      ArrayCopy(zipped, out, 0, 0, ArraySize(out));
   if(FlightClientSend("trades", zipped, ArraySize(zipped)))
      return(true);
   SocketErrors++;
   FallbackEvents++;
   trade_retry_count++;
   EnqueuePending(pending_trades, pending_trade_lines, zipped, line, log_dir + "/pending_trades.wal");
   TradeQueueDepth = ArraySize(pending_trades);
   datetime now = UseBrokerTime ? TimeCurrent() : TimeLocal();
   next_trade_flush = now + trade_backoff;
   return(false);
}

bool SendMetrics(uchar &payload[], string line)
{
   int len = ArraySize(payload);
   uchar out[];
   bool send_full = true;
   if(have_last_metric && ArraySize(last_metric_payload)==len)
   {
      int changes = 0;
      for(int i=0; i<len; i++)
         if(payload[i] != last_metric_payload[i])
            changes++;
      if(changes>0 && changes<256)
      {
         send_full = false;
         ArrayResize(out, 3 + changes*3);
         out[0] = (uchar)SCHEMA_VERSION;
         out[1] = 1;
         out[2] = (uchar)changes;
         int pos = 3;
         for(int i=0; i<len; i++)
         {
            if(payload[i] != last_metric_payload[i])
            {
               out[pos]   = (uchar)(i>>8);
               out[pos+1] = (uchar)(i & 0xFF);
               out[pos+2] = payload[i];
               pos += 3;
            }
         }
      }
   }
   if(send_full)
   {
      ArrayResize(out, len + 2);
      out[0] = (uchar)SCHEMA_VERSION;
      out[1] = 0;
      ArrayCopy(out, payload, 2, 0, len);
   }
   ArrayResize(last_metric_payload, len);
   ArrayCopy(last_metric_payload, payload, 0, 0, len);
   have_last_metric = true;
   uchar zipped[];
   if(!CryptEncode(CRYPT_ARCHIVE_GZIP, out, zipped))
      ArrayCopy(zipped, out, 0, 0, ArraySize(out));
   if(FlightClientSend("metrics", zipped, ArraySize(zipped)))
      return(true);
   SocketErrors++;
   FallbackEvents++;
   metric_retry_count++;
   EnqueuePending(pending_metrics, pending_metric_lines, zipped, line, log_dir + "/pending_metrics.wal");
   MetricQueueDepth = ArraySize(pending_metrics);
   datetime now = UseBrokerTime ? TimeCurrent() : TimeLocal();
   next_metric_flush = now + metric_backoff;
   return(false);
}

void FinalizeTradeEntry(PendingTrade &t, bool is_anom)
{
   int cal_id = CalendarEventIdAt(t.time_event);
   string line = StringFormat(
      "%d;%s;%s;%s;%s;%d;%d;%s;%s;%d;%.2f;%.5f;%.5f;%.5f;%.2f;%.2f;%.5f;%s;%s;%s;%.2f;%.5f;%d;%s;%.2f;%.2f;%.5f;%.5f;%.5f;%d;%d;%.2f;%.2f;%.2f;%.2f;%.2f;%.5f;%.5f;%s;%d;%d",
      t.id,
      TimeToString(t.time_event, TIME_DATE|TIME_SECONDS),
      TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
      TimeToString(TimeLocal(), TIME_DATE|TIME_SECONDS),
      t.action, t.ticket, t.magic, t.source, t.symbol, t.order_type,
      t.lots, t.price, t.sl, t.tp, t.profit, t.profit_after, t.spread,
      TraceId, t.span_id, t.comment_with_span, t.remaining, t.slippage,
      (int)t.volume, t.open_time_str, t.book_bid_vol, t.book_ask_vol,
      t.book_imbalance, t.sl_hit_dist, t.tp_hit_dist, t.decision_id, is_anom,
      t.equity, t.margin_level, t.commission, t.swap, t.risk_weight,
      t.trend_estimate, t.trend_variance, t.exit_reason, t.duration_sec, cal_id);

   uchar payload[];
   int len = SerializeTradeEvent(
      SCHEMA_VERSION, t.id, TraceId,
      TimeToString(t.time_event, TIME_DATE|TIME_SECONDS),
      TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
      TimeToString(TimeLocal(), TIME_DATE|TIME_SECONDS),
      t.action, t.ticket, t.magic, t.source, t.symbol, t.order_type,
      t.lots, t.price, t.sl, t.tp, t.profit, t.profit_after, t.spread, t.comment_with_span, t.remaining,
      t.slippage, (int)t.volume, t.open_time_str, t.book_bid_vol, t.book_ask_vol, t.book_imbalance, t.sl_hit_dist, t.tp_hit_dist, t.equity, t.margin_level, t.commission, t.swap, t.decision_id, t.exit_reason, t.duration_sec, payload);

   bool sent = false;
   if(len>0)
      sent = SendTrade(payload);
   if(!sent)
   {
      SocketErrors++;
      FallbackEvents++;
      if(CurrentBackend==LOG_BACKEND_CSV)
      {
         if(trade_log_handle==INVALID_HANDLE)
            return;
         if(EnableDebugLogging)
         {
            FileSeek(trade_log_handle, 0, SEEK_END);
            int _wr = FileWrite(trade_log_handle, line);
            if(_wr <= 0)
               FileWriteErrors++;
            FileFlush(trade_log_handle);
         }
         else
         {
            int n = ArraySize(trade_log_buffer);
            ArrayResize(trade_log_buffer, n+1);
            trade_log_buffer[n] = line;
         }
      }
      else if(CurrentBackend==LOG_BACKEND_SQLITE)
      {
         if(log_db_handle!=INVALID_HANDLE)
         {
            string sql = StringFormat(
               "INSERT INTO logs (event_id,event_time,broker_time,local_time,action,ticket,magic,source,symbol,order_type,lots,price,sl,tp,profit,profit_after_trade,spread,trace_id,span_id,comment,remaining_lots,slippage,volume,open_time,book_bid_vol,book_ask_vol,book_imbalance,sl_hit_dist,tp_hit_dist,decision_id,is_anomaly,equity,margin_level,commission,swap,exit_reason,duration_sec,calendar_event_id) VALUES (%d,'%s','%s','%s','%s',%d,%d,'%s','%s',%d,%.2f,%.5f,%.5f,%.5f,%.2f,%.2f,%d,'%s','%s','%s',%.2f,%.5f,%d,'%s',%.2f,%.2f,%.5f,%.5f,%.5f,%d,%d,%.2f,%.2f,%.2f,%.2f,'%s',%d,%d)",
               t.id,
               SqlEscape(TimeToString(t.time_event, TIME_DATE|TIME_SECONDS)),
               SqlEscape(TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS)),
               SqlEscape(TimeToString(TimeLocal(), TIME_DATE|TIME_SECONDS)),
               SqlEscape(t.action), t.ticket, t.magic, SqlEscape(t.source), SqlEscape(t.symbol), t.order_type,
               t.lots, t.price, t.sl, t.tp, t.profit, t.profit_after, t.spread, SqlEscape(TraceId), SqlEscape(t.span_id), SqlEscape(t.comment), t.remaining,
               t.slippage, (int)t.volume, SqlEscape(t.open_time_str), t.book_bid_vol, t.book_ask_vol, t.book_imbalance, t.sl_hit_dist, t.tp_hit_dist, t.decision_id, is_anom, t.equity, t.margin_level, t.commission, t.swap, SqlEscape(t.exit_reason), t.duration_sec, cal_id);
            DatabaseExecute(log_db_handle, sql);
         }
      }
   }
   SendOtelSpan(TraceId, t.span_id, t.action);
}

void ProcessAnomalyQueue()
{
   if(ArraySize(AnomalyQueue)==0)
      return;
   PendingTrade t = AnomalyQueue[0];
   datetime now = UseBrokerTime ? TimeCurrent() : TimeLocal();
   if(!t.anomaly_sent)
   {
      t.anomaly_id = GenId(8);
      if(CheckAnomaly(t.anomaly_id, t.price, t.sl, t.tp, t.lots, t.spread, t.slippage))
      {
         t.anomaly_sent = true;
         t.anomaly_sent_time = now;
         AnomalyQueue[0] = t;
      }
      else
      {
         t.anomaly_status = "enqueue_fail";
         t.comment_with_span = t.comment_with_span + ";anom_enqueue_fail";
         FinalizeTradeEntry(t, false);
         RemoveAnomaly(0);
      }
      return;
   }
   double err = 0.0;
   int res = PollAnomaly(t.anomaly_id, err);
   if(res==1)
   {
      bool is_anom = (err > AnomalyThreshold);
      if(now - t.anomaly_sent_time > AnomalyTimeoutSeconds)
         AnomalyLateResponses++;
      FinalizeTradeEntry(t, is_anom);
      RemoveAnomaly(0);
      return;
   }
   if(now - t.anomaly_sent_time > AnomalyTimeoutSeconds)
   {
      t.anomaly_status = "timeout";
      t.comment_with_span = t.comment_with_span + ";anom_timeout";
      FinalizeTradeEntry(t, false);
      RemoveAnomaly(0);
      return;
   }
   AnomalyQueue[0] = t;
}

string FileNameFromPath(string path)
{
   int pos = StringLen(path)-1;
   while(pos>=0)
   {
      ushort ch = StringGetChar(path, pos);
      if(ch=='\\' || ch=='/')
         break;
      pos--;
   }
   return(StringSubstr(path, pos+1));
}

bool ComputeFileSHA256(string filename, string &hash)
{
   int h = FileOpen(filename, FILE_READ|FILE_BIN|FILE_SHARE_READ);
   if(h==INVALID_HANDLE)
      return(false);
   int size = (int)FileSize(h);
   uchar data[];
   ArrayResize(data, size);
   FileReadArray(h, data, 0, size);
   FileClose(h);
   uchar result[];
   if(!CryptEncode(CRYPT_HASH_SHA256, data, result))
      return(false);
   hash = "";
   for(int i=0; i<ArraySize(result); i++)
      hash += StringFormat("%02x", result[i]);
   return(true);
}

string SqlEscape(string s)
{
   return(StringReplace(s, "'", "''"));
}

void LogTrade(int event_id, string action, int ticket, int magic, string source,
              string symbol, int order_type, double lots, double price,
              double req_price, double sl, double tp, double profit,
              double profit_after, double remaining, datetime time_event,
              string comment, double volume, datetime open_time,
              double book_bid_vol, double book_ask_vol, double book_imbalance,
              double slippage, double equity, double margin_level,
              double risk_weight)
  {
   PendingTrade t;
   t.id = event_id;
   string trace_id = "";
   string span_id = "";
   int pos = StringFind(comment, "trace_id=");
   if(pos >= 0)
   {
      int start = pos + StringLen("trace_id=");
      int end = start;
      while(end < StringLen(comment))
      {
         string ch = StringMid(comment, end, 1);
         if(ch==";" || ch=="|" )
            break;
         end++;
      }
      trace_id = StringSubstr(comment, start, end-start);
   }
   pos = StringFind(comment, "span_id=");
   if(pos >= 0)
   {
      int start = pos + StringLen("span_id=");
      int end = start;
      while(end < StringLen(comment))
      {
         string ch = StringMid(comment, end, 1);
         if(ch==";" || ch=="|" )
            break;
         end++;
      }
      span_id = StringSubstr(comment, start, end-start);
   }
   if(StringLen(trace_id)==0)
      trace_id = GenId(16);
   TraceId = trace_id;
   if(StringLen(span_id)==0)
      span_id = GenId(8);
   t.span_id = span_id;
   t.action = action;
   t.ticket = ticket;
   t.magic = magic;
   t.source = source;
   t.symbol = symbol;
   t.order_type = order_type;
   t.lots = lots;
   t.price = price;
   t.req_price = req_price;
   t.sl = sl;
   t.tp = tp;
   t.profit = profit;
   t.profit_after = profit_after;
   t.remaining = remaining;
   t.time_event = time_event;
   t.comment = comment;
   t.volume = volume;
   t.open_time = open_time;
   t.book_bid_vol = book_bid_vol;
   t.book_ask_vol = book_ask_vol;
   t.book_imbalance = book_imbalance;
   t.spread = MarketInfo(symbol, MODE_ASK) - MarketInfo(symbol, MODE_BID);
   t.slippage = slippage;
   t.equity = equity;
   t.margin_level = margin_level;
   t.commission = 0.0;
   t.swap = 0.0;
   if(action=="CLOSE")
   {
      t.commission = OrderCommission();
      t.swap = OrderSwap();
   }
   t.risk_weight = risk_weight;
   t.trend_estimate = trend_estimate;
    t.trend_variance = trend_variance;
   if(StringFind(comment, "span=") < 0 && StringFind(comment, "span_id=") < 0)
   {
      string span_comment = "span=" + t.span_id;
      if(StringLen(comment) > 0)
         span_comment += ";" + comment;
      t.comment_with_span = span_comment;
   }
   else
      t.comment_with_span = comment;
   // Extract decision id if present in the order comment
   int decision_id = 0;
   int pos = StringFind(comment, "decision_id=");
   if(pos >= 0)
   {
      int start = pos + StringLen("decision_id=");
      int end = start;
      while(end < StringLen(comment))
      {
         string ch = StringMid(comment, end, 1);
         if(ch==";" || ch=="|")
            break;
         end++;
      }
      decision_id = (int)StringToInteger(StringSubstr(comment, start, end-start));
   }
   t.decision_id = decision_id;
   string open_time_str = "";
   if(action=="CLOSE" && open_time>0)
      open_time_str = TimeToString(open_time, TIME_DATE|TIME_SECONDS);
   t.open_time_str = open_time_str;
   double sl_hit_dist = 0.0;
   double tp_hit_dist = 0.0;
   if(action=="CLOSE")
   {
      if(sl!=0.0)
         sl_hit_dist = MathAbs(price - sl);
      if(tp!=0.0)
         tp_hit_dist = MathAbs(price - tp);
   }
   t.sl_hit_dist = sl_hit_dist;
   t.tp_hit_dist = tp_hit_dist;
   string exit_reason = "";
   int duration_sec = 0;
   if(action=="CLOSE")
   {
      double pt = MarketInfo(symbol, MODE_POINT);
      exit_reason = "MANUAL";
      if(tp!=0.0 && MathAbs(price - tp) <= pt*2)
         exit_reason = "TP";
      else if(sl!=0.0 && MathAbs(price - sl) <= pt*2)
         exit_reason = "SL";
      if(open_time>0)
         duration_sec = (int)(time_event - open_time);
   }
   t.exit_reason = exit_reason;
   t.duration_sec = duration_sec;
   t.start_time = UseBrokerTime ? TimeCurrent() : TimeLocal();
   EnqueueAnomaly(t);
}


void ExportLogs(datetime ts)
{
   string src = log_dir + "/trades_raw.csv";
   if(!FileIsExist(src))
      return;
   string ts_str = TimeToString(ts, TIME_DATE|TIME_MINUTES);
   ts_str = StringReplace(ts_str, ".", "-");
   ts_str = StringReplace(ts_str, ":", "");
   ts_str = StringReplace(ts_str, " ", "_");
   int last_id = NextEventId - 1;
   string base = log_dir + "/" + ts_str + "_" + IntegerToString(last_id);
   string dest = base + ".csv";
   int in_h = FileOpen(src, FILE_CSV|FILE_READ|FILE_TXT|FILE_SHARE_READ|FILE_SHARE_WRITE, ';');
   if(in_h==INVALID_HANDLE)
      return;
   string header = "";
   string lines[];
   datetime start_time = 0;
   datetime end_time = 0;
   if(!FileIsEnding(in_h))
      header = FileReadString(in_h);
   string parts[];
   while(!FileIsEnding(in_h))
   {
      string line = FileReadString(in_h);
      if(StringLen(line)==0)
         continue;
      int n = ArraySize(lines);
      ArrayResize(lines, n+1);
      lines[n] = line;
      int cnt = StringSplit(line, ';', parts);
      if(cnt>1)
      {
         datetime ev = StringToTime(parts[1]);
         if(start_time==0)
            start_time = ev;
         end_time = ev;
      }
   }
   FileClose(in_h);
   int out_h = FileOpen(dest, FILE_CSV|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE, ';');
   if(out_h!=INVALID_HANDLE)
   {
      string meta = StringFormat("{\"schema_version\":%d,\"commit\":\"%s\",\"start_time\":\"%s\",\"end_time\":\"%s\"}",
                               LogSchemaVersion, CommitHash,
                               TimeToString(start_time, TIME_DATE|TIME_SECONDS),
                               TimeToString(end_time, TIME_DATE|TIME_SECONDS));
      int _wr = FileWrite(out_h, meta);
      if(_wr <= 0) FileWriteErrors++;
      _wr = FileWrite(out_h, header);
      if(_wr <= 0) FileWriteErrors++;
      for(int i=0; i<ArraySize(lines); i++)
      {
         _wr = FileWrite(out_h, lines[i]);
         if(_wr <= 0) FileWriteErrors++;
      }
   FileClose(out_h);
   }
   FileDelete(src);

   FileCopy(ModelStateFile, base + ".state.json", FILE_COMMON);

   // Compress the rotated file and remove the original
   int rh = FileOpen(dest, FILE_READ|FILE_BIN);
   if(rh!=INVALID_HANDLE)
   {
      int size = (int)FileSize(rh);
      uchar raw[];
      ArrayResize(raw, size);
      FileReadArray(rh, raw, 0, size);
      FileClose(rh);
      uchar zipped[];
      if(CryptEncode(CRYPT_ARCHIVE_GZIP, raw, zipped))
      {
         string gz_dest = dest + ".gz";
         int zh = FileOpen(gz_dest, FILE_WRITE|FILE_BIN);
         if(zh!=INVALID_HANDLE)
         {
            FileWriteArray(zh, zipped, 0, WHOLE_ARRAY);
            FileClose(zh);
            FileDelete(dest);
            dest = gz_dest;
         }
      }
   }

   string checksum;
   if(ComputeFileSHA256(dest, checksum))
   {
      string manifest = dest + ".manifest.json";
      int mh = FileOpen(manifest, FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE);
      if(mh!=INVALID_HANDLE)
      {
         string fname = FileNameFromPath(dest);
         string mjson = StringFormat("{\"file\":\"%s\",\"checksum\":\"%s\",\"commit\":\"%s\"}",
                                    fname, checksum, CommitHash);
         int _wr = FileWrite(mh, mjson);
         if(_wr <= 0) FileWriteErrors++;
         FileClose(mh);
      }
   }
}

void WriteMetrics(datetime ts)
{
   int h = INVALID_HANDLE;
   string fname = log_dir + "/metrics.csv";

   datetime cutoff = ts - MetricsRollingDays*24*60*60;
   int trade_q_depth = TradeQueueDepth;
   int metric_q_depth = MetricQueueDepth;
   int wal_size = 0;
   string tw = log_dir + "/pending_trades.wal";
   string mw = log_dir + "/pending_metrics.wal";
   if(FileIsExist(tw))
      wal_size += (int)FileSize(tw);
   if(FileIsExist(mw))
      wal_size += (int)FileSize(mw);
   for(int m=0; m<ArraySize(target_magics); m++)
   {
      int magic = target_magics[m];
      int trades = 0;
      int wins = 0;
      int losses = 0;
      double profit_total = 0.0;
      double sum_profit = 0.0;
      double sum_sq_profit = 0.0;
      double neg_sum = 0.0;
      double neg_sum_sq = 0.0;
      double loss_sum = 0.0;
      double cumulative = 0.0;
      double peak = 0.0;
      double max_dd = 0.0;
      double profits[];

      for(int i=0; i<OrdersHistoryTotal(); i++)
      {
         if(!OrderSelect(i, SELECT_BY_POS, MODE_HISTORY))
            continue;
         if(OrderMagicNumber()!=magic)
            continue;
         if(OrderCloseTime() < cutoff)
            continue;

         double p = OrderProfit()+OrderSwap()+OrderCommission();
         trades++;
         if(p >= 0)
            wins++;
         else
         {
            losses++;
            neg_sum += p;
            neg_sum_sq += p*p;
            loss_sum += -p;
         }
         int pidx = ArraySize(profits);
         ArrayResize(profits, pidx+1);
         profits[pidx] = p;
         profit_total += p;
         sum_profit += p;
         sum_sq_profit += p*p;
         cumulative += p;
         if(cumulative > peak)
            peak = cumulative;
         double dd = peak - cumulative;
         if(dd > max_dd)
            max_dd = dd;
      }

      if(trades==0)
         continue;

      double win_rate = double(wins)/trades;
      double avg_profit = profit_total/trades;
      double avg_loss = losses>0 ? loss_sum/losses : 0.0;
      double variance = trades>0 ? sum_sq_profit/trades - MathPow(sum_profit/trades, 2) : 0.0;
      double stddev = variance>0 ? MathSqrt(variance) : 0.0;
      double sharpe = stddev>0 ? (sum_profit/trades)/stddev : 0.0;
      double var_neg = losses>0 ? neg_sum_sq/losses - MathPow(neg_sum/losses, 2) : 0.0;
      double stddev_neg = var_neg>0 ? MathSqrt(var_neg) : 0.0;
      double sortino = stddev_neg>0 ? (sum_profit/trades)/stddev_neg : 0.0;
      double expectancy = (avg_profit * win_rate) - (avg_loss * (1.0 - win_rate));

      ArraySort(profits, WHOLE_ARRAY, 0, MODE_ASCEND);
      int vidx = (int)MathFloor(0.05 * ArraySize(profits));
      if(vidx < 0) vidx = 0;
      if(vidx >= ArraySize(profits)) vidx = ArraySize(profits)-1;
      double var95 = profits[vidx];
      int var_breach_count = 0;
      for(int vb=0; vb<ArraySize(profits); vb++)
         if(profits[vb] < var95) var_breach_count++;

      string span_id = GenId(8);
      int fallback_flag = (trade_retry_count >= FallbackRetryThreshold || metric_retry_count >= FallbackRetryThreshold) ? 1 : 0;
      int anomaly_pending = AnomalyQueueDepth;
      int anomaly_late = AnomalyLateResponses;
      // emit file/socket errors and retry counts for monitoring
      string line = StringFormat("%s;%d;%.3f;%.2f;%d;%.2f;%.3f;%.3f;%.2f;%d;%d;%.2f;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%s;%s", TimeToString(ts, TIME_DATE|TIME_MINUTES), magic, win_rate, avg_profit, trades, max_dd, sharpe, sortino, expectancy, FileWriteErrors, SocketErrors, CpuLoad, CachedBookRefreshSeconds, var_breach_count, trade_q_depth, metric_q_depth, FallbackEvents, fallback_flag, wal_size, trade_retry_count, metric_retry_count, anomaly_pending, anomaly_late, TraceId, span_id);

      uchar payload[];
      int len = SerializeMetrics(
         SCHEMA_VERSION,
         TimeToString(ts, TIME_DATE|TIME_MINUTES), magic, win_rate, avg_profit,
         trades, max_dd, sharpe, FileWriteErrors, SocketErrors, CpuLoad, CachedBookRefreshSeconds, var_breach_count, trade_q_depth, metric_q_depth, FallbackEvents, fallback_flag, wal_size, trade_retry_count, metric_retry_count, anomaly_pending, anomaly_late, payload);
      bool sent = false;
      if(len>0)
         sent = SendMetrics(payload, line);
      if(!sent)
      {
         FallbackEvents++;
         if(h==INVALID_HANDLE)
         {
            h = FileOpen(fname, FILE_CSV|FILE_READ|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ, ';');
            if(h==INVALID_HANDLE)
            {
               h = FileOpen(fname, FILE_CSV|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE, ';');
               if(h!=INVALID_HANDLE)
               {
                  int _wr = FileWrite(h, "time;magic;win_rate;avg_profit;trade_count;drawdown;sharpe;sortino;expectancy;file_write_errors;socket_errors;cpu_load;book_refresh_seconds;var_breach_count;trade_queue_depth;metric_queue_depth;fallback_events;fallback_logging;wal_size;trade_retry_count;metric_retry_count;anomaly_pending;anomaly_late;trace_id;span_id");
                  if(_wr <= 0)
                     FileWriteErrors++;
               }
               else
               {
                  FileWriteErrors++;
               }
            }
            else
               FileSeek(h, 0, SEEK_END);
         }
         if(h!=INVALID_HANDLE)
         {
            int _wr_line = FileWrite(h, line);
            if(_wr_line <= 0)
               FileWriteErrors++;
         }
      }
      SendOtelSpan(TraceId, span_id, "metrics");
   }

   if(h!=INVALID_HANDLE)
      FileClose(h);
}

void UpdateModelMetrics(datetime ts)
{
   string fname = log_dir + "/metrics.csv";
   int h = FileOpen(fname, FILE_CSV|FILE_READ|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ, ';');
   if(h==INVALID_HANDLE)
   {
      h = FileOpen(fname, FILE_CSV|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE, ';');
      if(h==INVALID_HANDLE)
         return;
      int _wr_h = FileWrite(h, "time;model_id;hit_rate;profit_factor");
      if(_wr_h <= 0)
         FileWriteErrors++;
   }
   else
   {
      FileSeek(h, 0, SEEK_END);
   }

   datetime cutoff = ts - MetricsRollingDays*24*60*60;
   for(int m=0; m<ArraySize(target_magics); m++)
   {
      int magic = target_magics[m];
      int trades = 0;
      int wins = 0;
      double profit_pos = 0.0;
      double profit_neg = 0.0;

      for(int i=OrdersHistoryTotal()-1; i>=0; i--)
      {
         if(!OrderSelect(i, SELECT_BY_POS, MODE_HISTORY))
            continue;
         if(OrderMagicNumber()!=magic)
            continue;
         if(OrderCloseTime() < cutoff)
            continue;

         double p = OrderProfit()+OrderSwap()+OrderCommission();
         trades++;
         if(p >= 0)
         {
            wins++;
            profit_pos += p;
         }
         else
         {
            profit_neg += -p;
         }
      }

      if(trades==0)
         continue;

      double hit_rate = double(wins)/trades;
      double profit_factor = profit_neg>0 ? profit_pos/profit_neg : 0.0;

      string line = StringFormat("%s;%d;%.3f;%.3f", TimeToString(ts, TIME_DATE|TIME_MINUTES), magic, hit_rate, profit_factor);
      int _wr_line = FileWrite(h, line);
      if(_wr_line <= 0)
         FileWriteErrors++;
   }

   FileClose(h);
}

void ManageMetrics(datetime ts)
{
   string fname = log_dir + "/metrics.csv";
   if(!FileIsExist(fname))
      return;
   int in_h = FileOpen(fname, FILE_CSV|FILE_READ|FILE_TXT|FILE_SHARE_READ|FILE_SHARE_WRITE, ';');
   if(in_h==INVALID_HANDLE)
      return;
   string lines[];
   datetime cutoff = ts - MetricsDaysToKeep*24*60*60;
   while(!FileIsEnding(in_h))
   {
      string l = FileReadString(in_h);
      if(StringLen(l)==0)
         continue;
      string parts[];
      int c = StringSplit(l, ';', parts);
      datetime t = 0;
      if(c>0)
         t = StringToTime(parts[0]);
      if(t>=cutoff || t==0)
      {
         int n = ArraySize(lines);
         ArrayResize(lines, n+1);
         lines[n] = l;
      }
   }
   FileClose(in_h);

   int out_h = FileOpen(fname, FILE_CSV|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE, ';');
   if(out_h==INVALID_HANDLE)
      return;
  for(int i=0; i<ArraySize(lines); i++)
  {
     int _wr = FileWrite(out_h, lines[i]);
     if(_wr <= 0)
        FileWriteErrors++;
  }
  FileClose(out_h);
}

void FlushTradeBuffer()
{
   if(CurrentBackend!=LOG_BACKEND_CSV)
      return;
   if(trade_log_handle==INVALID_HANDLE)
      return;
   int n = ArraySize(trade_log_buffer);
   if(n==0)
      return;
   FileSeek(trade_log_handle, 0, SEEK_END);
   for(int i=0; i<n; i++)
   {
      int _wr = FileWrite(trade_log_handle, trade_log_buffer[i]);
      if(_wr <= 0)
         FileWriteErrors++;
   }
   FileFlush(trade_log_handle);
   ArrayResize(trade_log_buffer, 0);
   SaveModelState();
}

void FlushMetricBuffer()
{
   SaveModelState();
}
