#property strict
#include "model_interface.mqh"
#include <Arrays/ArrayInt.mqh>

#import "observer_proto.dll"
int SerializeTradeEvent(int schema_version, int event_id, string trace_id, string event_time, string broker_time, string local_time, string action, int ticket, int magic, string source, string symbol, int order_type, double lots, double price, double sl, double tp, double profit, double profit_after_trade, double spread, string comment, double remaining_lots, double slippage, int volume, string open_time, double book_bid_vol, double book_ask_vol, double book_imbalance, double sl_hit_dist, double tp_hit_dist, int decision_id, uchar &out[]);
int SerializeMetrics(int schema_version, string time, int magic, double win_rate, double avg_profit, int trade_count, double drawdown, double sharpe, int file_write_errors, int socket_errors, int book_refresh_seconds, uchar &out[]);
#import

#import "flight_client.dll"
int FlightConnect(string address);
bool FlightSendTrade(int client, uchar &payload[], int len);
bool FlightSendMetrics(int client, uchar &payload[], int len);
void FlightClose(int client);
#import

extern string TargetMagicNumbers = "12345,23456";
extern int    LearningExportIntervalMinutes = 15;
extern int    PredictionWindowSeconds       = 60;
extern double LotSizeTolerancePct           = 20.0;
extern double PriceTolerancePips            = 5.0;
extern bool   EnableLiveCloneMode           = false;
extern int    MaxModelsToRetain             = 3;
extern int    MetricsRollingDays            = 7;
extern int    MetricsDaysToKeep             = 30;
extern string LogDirectoryName              = "observer_logs"; // resume event_id from existing logs, start at 1 if none
extern bool   EnableDebugLogging            = false;
extern bool   UseBrokerTime                 = true;
extern string SymbolsToTrack                = ""; // empty=all
extern string FlightServerUri              = "grpc://127.0.0.1:8815";
extern string CommitHash                   = "";
extern string ModelVersion                 = "";
extern string TraceId                      = "";
extern int    BookRefreshSeconds           = 5;
extern string AnomalyServiceUrl            = "http://127.0.0.1:8000/anomaly";
extern double AnomalyThreshold             = 0.1;
extern string OtelEndpoint                = "";

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
int      FileWriteErrors = 0;
int      FlightClient = INVALID_HANDLE;
int      FlightErrors = 0;
const int LogSchemaVersion = 3;

double   CpuLoad = 0.0;
int      CachedBookRefreshSeconds = 0;

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

bool CheckAnomaly(double price, double sl, double tp, double lots, double spread, double slippage)
{
   string payload = StringFormat("[%.5f,%.5f,%.5f,%.2f,%.5f,%.5f]", price, sl, tp, lots, spread, slippage);
   uchar data[];
   StringToCharArray(payload, data);
   uchar result[];
   string headers = "Content-Type: application/json";
   string rheaders = "";
   int res = WebRequest("POST", AnomalyServiceUrl, headers, 5000, data, ArraySize(data)-1, result, rheaders);
   if(res==200)
   {
      string txt = CharArrayToString(result);
      double err = StrToDouble(txt);
      return(err > AnomalyThreshold);
   }
   return(false);
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

   DirectoryCreate(LogDirectoryName);
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
   int info_handle = FileOpen(LogDirectoryName + "\\run_info.json", FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ);
   if(info_handle!=INVALID_HANDLE)
   {
      FileWriteString(info_handle, run_info);
      FileClose(info_handle);
   }

   CachedBookRefreshSeconds = BookRefreshSeconds;
   FlightClient = FlightConnect(FlightServerUri);

   // Prefer SQLite backend, fall back to CSV
   string db_fname = LogDirectoryName + "\\trades_raw.sqlite";
   log_db_handle = DatabaseOpen(db_fname, DATABASE_OPEN_READWRITE|DATABASE_OPEN_CREATE);
   if(log_db_handle!=INVALID_HANDLE)
   {
      string create_sql = "CREATE TABLE IF NOT EXISTS logs (event_id INTEGER, event_time TEXT, broker_time TEXT, local_time TEXT, action TEXT, ticket INTEGER, magic INTEGER, source TEXT, symbol TEXT, order_type INTEGER, lots REAL, price REAL, sl REAL, tp REAL, profit REAL, profit_after_trade REAL, spread INTEGER, comment TEXT, remaining_lots REAL, slippage REAL, volume INTEGER, open_time TEXT, book_bid_vol REAL, book_ask_vol REAL, book_imbalance REAL, sl_hit_dist REAL, tp_hit_dist REAL, decision_id INTEGER, is_anomaly INTEGER)";
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
      string log_fname = LogDirectoryName + "\\trades_raw.csv";
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
            string header = "event_id;event_time;broker_time;local_time;action;ticket;magic;source;symbol;order_type;lots;price;sl;tp;profit;profit_after_trade;spread;comment;remaining_lots;slippage;volume;open_time;book_bid_vol;book_ask_vol;book_imbalance;sl_hit_dist;tp_hit_dist;decision_id;is_anomaly";
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
   if(FlightClient!=INVALID_HANDLE)
   {
      FlightClose(FlightClient);
      FlightClient = INVALID_HANDLE;
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

   if(entry==DEAL_ENTRY_IN || entry==DEAL_ENTRY_INOUT)
   {
      double bid_vol, ask_vol, book_imb;
      GetBookVolumes(symbol, bid_vol, ask_vol, book_imb);
      LogTrade("OPEN", ticket, magic, "mt4", symbol, order_type,
               lots, price, req.price, sl, tp, 0.0, profit_after,
               remaining, now, comment, iVolume(symbol, 0, 0), 0,
               bid_vol, ask_vol, book_imb);
      if(!IsTracked(ticket))
         AddTicket(ticket);
      else if(entry==DEAL_ENTRY_INOUT && remaining>0.0 && OrderSelect(ticket, SELECT_BY_TICKET, MODE_TRADES))
      {
         double cur_price = OrderOpenPrice();
         double cur_sl    = OrderStopLoss();
         double cur_tp    = OrderTakeProfit();
         double bid_vol2, ask_vol2, book_imb2;
         GetBookVolumes(symbol, bid_vol2, ask_vol2, book_imb2);
         LogTrade("MODIFY", ticket, magic, "mt4", symbol, order_type,
                  0.0, cur_price, cur_price, cur_sl, cur_tp, 0.0, profit_after,
                  remaining, now, comment, iVolume(symbol, 0, 0), 0,
                  bid_vol2, ask_vol2, book_imb2);
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
      LogTrade("CLOSE", ticket, magic, "mt4", symbol, order_type,
               lots, price, req.price, sl, tp, profit, profit_after,
               remaining, now, comment, iVolume(symbol, 0, 0), open_time,
               bid_vol3, ask_vol3, book_imb3);
      if(IsTracked(ticket) && remaining==0.0)
         RemoveTicket(ticket);
      else if(remaining>0.0 && OrderSelect(ticket, SELECT_BY_TICKET, MODE_TRADES))
      {
         double cur_price = OrderOpenPrice();
         double cur_sl    = OrderStopLoss();
         double cur_tp    = OrderTakeProfit();
         double bid_vol4, ask_vol4, book_imb4;
         GetBookVolumes(symbol, bid_vol4, ask_vol4, book_imb4);
         LogTrade("MODIFY", ticket, magic, "mt4", symbol, order_type,
                  0.0, cur_price, cur_price, cur_sl, cur_tp, 0.0, profit_after,
                  remaining, now, comment, iVolume(symbol, 0, 0), 0,
                  bid_vol4, ask_vol4, book_imb4);
      }
   }
}

void OnTick()
{
   uint tick_start = GetTickCount();
   datetime now = UseBrokerTime ? TimeCurrent() : TimeLocal();
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
         LogTrade("OPEN", ticket, OrderMagicNumber(), "mt4", OrderSymbol(), OrderType(),
                  OrderLots(), OrderOpenPrice(), OrderOpenPrice(), OrderStopLoss(), OrderTakeProfit(),
                  0.0, profit_after, OrderLots(), now, OrderComment(),
                  iVolume(OrderSymbol(), 0, 0), 0, bid_vol, ask_vol, book_imb);
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
             LogTrade("CLOSE", ticket, OrderMagicNumber(), "mt4", OrderSymbol(),
                       OrderType(), OrderLots(), OrderClosePrice(), OrderClosePrice(), OrderStopLoss(),
                       OrderTakeProfit(), OrderProfit()+OrderSwap()+OrderCommission(),
                       profit_after2, 0.0, now, OrderComment(),
                       iVolume(OrderSymbol(), 0, 0), OrderOpenTime(),
                       bid_vol, ask_vol, book_imb);
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
   FlushTradeBuffer();
   datetime now = UseBrokerTime ? TimeCurrent() : TimeLocal();

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

void SendTrade(uchar &payload[])
{
   if(FlightClient==INVALID_HANDLE)
      return;
   int len = ArraySize(payload);
   if(!FlightSendTrade(FlightClient, payload, len))
      FlightErrors++;
}

void SendMetrics(uchar &payload[])
{
   if(FlightClient==INVALID_HANDLE)
      return;
   int len = ArraySize(payload);
   if(!FlightSendMetrics(FlightClient, payload, len))
      FlightErrors++;
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

void LogTrade(string action, int ticket, int magic, string source,
              string symbol, int order_type, double lots, double price,
              double req_price, double sl, double tp, double profit,
              double profit_after, double remaining, datetime time_event,
              string comment, double volume, datetime open_time,
              double book_bid_vol, double book_ask_vol, double book_imbalance)
{
   int id = NextEventId++;
   string span_id = GenId(8);
   string comment_with_span = "span=" + span_id;
   if(StringLen(comment) > 0)
      comment_with_span += ";" + comment;
   int decision_id = 0;
   int pos = StringFind(comment, "decision_id=");
   if(pos >= 0)
   {
      int start = pos + StringLen("decision_id=");
      int end = start;
      while(end < StringLen(comment))
      {
         string ch = StringMid(comment, end, 1);
         if(ch < "0" || ch > "9")
            break;
         end++;
      }
      decision_id = (int)StringToInteger(StringSubstr(comment, start, end-start));
   }
   string open_time_str = "";
   if(action=="CLOSE" && open_time>0)
      open_time_str = TimeToString(open_time, TIME_DATE|TIME_SECONDS);
   double sl_hit_dist = 0.0;
   double tp_hit_dist = 0.0;
   if(action=="CLOSE")
   {
      if(sl!=0.0)
         sl_hit_dist = MathAbs(price - sl);
      if(tp!=0.0)
         tp_hit_dist = MathAbs(price - tp);
   }
   double spread = MarketInfo(symbol, MODE_ASK) - MarketInfo(symbol, MODE_BID);
   double slippage = price - req_price;
   bool is_anom = CheckAnomaly(price, sl, tp, lots, spread, slippage);

   string line = StringFormat(
      "%d;%s;%s;%s;%s;%d;%d;%s;%s;%d;%.2f;%.5f;%.5f;%.5f;%.2f;%.2f;%.5f;%s;%.2f;%.5f;%d;%s;%.2f;%.2f;%.5f;%.5f;%.5f;%d;%d",
      id,
      TimeToString(time_event, TIME_DATE|TIME_SECONDS),
      TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
      TimeToString(TimeLocal(), TIME_DATE|TIME_SECONDS),
      action, ticket, magic, source, symbol, order_type, lots, price, sl, tp,
      profit, profit_after, spread, comment_with_span, remaining, slippage, (int)volume,
      open_time_str, book_bid_vol, book_ask_vol, book_imbalance, sl_hit_dist, tp_hit_dist, decision_id, is_anom);

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
            "INSERT INTO logs (event_id,event_time,broker_time,local_time,action,ticket,magic,source,symbol,order_type,lots,price,sl,tp,profit,profit_after_trade,spread,comment,remaining_lots,slippage,volume,open_time,book_bid_vol,book_ask_vol,book_imbalance,sl_hit_dist,tp_hit_dist,decision_id,is_anomaly) VALUES (%d,'%s','%s','%s','%s',%d,%d,'%s','%s',%d,%.2f,%.5f,%.5f,%.5f,%.2f,%.2f,%d,'%s',%.2f,%.5f,%d,'%s',%.2f,%.2f,%.5f,%.5f,%.5f,%d,%d)",
            id,
            SqlEscape(TimeToString(time_event, TIME_DATE|TIME_SECONDS)),
            SqlEscape(TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS)),
            SqlEscape(TimeToString(TimeLocal(), TIME_DATE|TIME_SECONDS)),
            SqlEscape(action), ticket, magic, SqlEscape(source), SqlEscape(symbol), order_type,
            lots, price, sl, tp, profit, profit_after, spread, SqlEscape(comment), remaining,
            slippage, (int)volume, SqlEscape(open_time_str), book_bid_vol, book_ask_vol, book_imbalance, sl_hit_dist, tp_hit_dist, decision_id, is_anom);
         DatabaseExecute(log_db_handle, sql);
      }
   }

   uchar payload[];
   int len = SerializeTradeEvent(
      LogSchemaVersion, id, TraceId,
      TimeToString(time_event, TIME_DATE|TIME_SECONDS),
      TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
      TimeToString(TimeLocal(), TIME_DATE|TIME_SECONDS),
      action, ticket, magic, source, symbol, order_type,
      lots, price, sl, tp, profit, profit_after, spread, comment_with_span, remaining,
      slippage, (int)volume, open_time_str, book_bid_vol, book_ask_vol, book_imbalance, sl_hit_dist, tp_hit_dist, decision_id, payload);
   if(len>0)
      SendTrade(payload);
   SendOtelSpan(TraceId, span_id, action);
}


void ExportLogs(datetime ts)
{
   string src = LogDirectoryName + "\\trades_raw.csv";
   if(!FileIsExist(src))
      return;
   string ts_str = TimeToString(ts, TIME_DATE|TIME_MINUTES);
   ts_str = StringReplace(ts_str, ".", "-");
   ts_str = StringReplace(ts_str, ":", "");
   ts_str = StringReplace(ts_str, " ", "_");
   int last_id = NextEventId - 1;
   string dest = LogDirectoryName + "\\" + ts_str + "_" + IntegerToString(last_id) + ".csv";
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
   string fname = LogDirectoryName + "\\metrics.csv";
   h = FileOpen(fname, FILE_CSV|FILE_READ|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ, ';');
   if(h==INVALID_HANDLE)
   {
      h = FileOpen(fname, FILE_CSV|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE, ';');
      if(h==INVALID_HANDLE)
         return;
      int _wr = FileWrite(h, "time;magic;win_rate;avg_profit;trade_count;drawdown;sharpe;sortino;expectancy;file_write_errors;socket_errors;book_refresh_seconds;trace_id;span_id");
      if(_wr <= 0)
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

      string span_id = GenId(8);
      string line = StringFormat("%s;%d;%.3f;%.2f;%d;%.2f;%.3f;%.3f;%.2f;%d;%d;%d;%s;%s", TimeToString(ts, TIME_DATE|TIME_MINUTES), magic, win_rate, avg_profit, trades, max_dd, sharpe, sortino, expectancy, FileWriteErrors, FlightErrors, CachedBookRefreshSeconds, TraceId, span_id);
      if(h!=INVALID_HANDLE)
      {
         int _wr_line = FileWrite(h, line);
         if(_wr_line <= 0)
            FileWriteErrors++;
      }

       uchar payload[];
       int len = SerializeMetrics(
         LogSchemaVersion,
         TimeToString(ts, TIME_DATE|TIME_MINUTES), magic, win_rate, avg_profit,
         trades, max_dd, sharpe, FileWriteErrors, FlightErrors, CachedBookRefreshSeconds, payload);
       if(len>0)
         SendMetrics(payload);
      SendOtelSpan(TraceId, span_id, "metrics");
   }

   if(h!=INVALID_HANDLE)
      FileClose(h);
}

void UpdateModelMetrics(datetime ts)
{
   string fname = LogDirectoryName + "\\metrics.csv";
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
   string fname = LogDirectoryName + "\\metrics.csv";
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
}
