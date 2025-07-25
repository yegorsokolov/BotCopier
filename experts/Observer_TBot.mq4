#property strict
#include "model_interface.mqh"
#include <Arrays/ArrayInt.mqh>

extern string TargetMagicNumbers = "12345,23456";
extern int    LearningExportIntervalMinutes = 15;
extern int    PredictionWindowSeconds       = 60;
extern double LotSizeTolerancePct           = 20.0;
extern double PriceTolerancePips            = 5.0;
extern bool   EnableLiveCloneMode           = false;
extern int    MaxModelsToRetain             = 3;
extern int    MetricsRollingDays            = 7;
extern int    MetricsDaysToKeep             = 30;
extern string LogDirectoryName              = "observer_logs";
extern bool   EnableDebugLogging            = false;
extern bool   UseBrokerTime                 = true;
extern string SymbolsToTrack                = ""; // empty=all
extern bool   EnableSocketLogging           = false;
extern string LogSocketHost                 = "127.0.0.1";
extern int    LogSocketPort                 = 9000;
extern int    LogBufferSize                 = 10;

int timer_handle;

int      tracked_tickets[];
CArrayInt ticket_map;
int      target_magics[];
string   track_symbols[];
datetime last_export = 0;
int      trade_log_handle = INVALID_HANDLE;
int      log_socket = INVALID_HANDLE;
datetime last_socket_attempt = 0;
string   trade_log_buffer[];
int      NextEventId = 1;

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

int OnInit()
{
   EventSetTimer(1);
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

   string log_fname = LogDirectoryName + "\\trades_raw.csv";
   trade_log_handle = FileOpen(log_fname, FILE_CSV|FILE_WRITE|FILE_READ|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ, ';');
   if(trade_log_handle!=INVALID_HANDLE)
   {
      bool need_header = (FileSize(trade_log_handle)==0);
      FileSeek(trade_log_handle, 0, SEEK_END);
      if(need_header)
      {
         string header = "event_id;event_time;broker_time;local_time;action;ticket;magic;source;symbol;order_type;lots;price;sl;tp;profit;comment;remaining_lots";
         FileWrite(trade_log_handle, header);
      }
   }

   if(EnableSocketLogging)
   {
      last_socket_attempt = UseBrokerTime ? TimeCurrent() : TimeLocal();
      log_socket = SocketCreate();
      if(log_socket!=INVALID_HANDLE)
      {
         if(!SocketConnect(log_socket, LogSocketHost, LogSocketPort, 1000))
         {
            if(EnableDebugLogging)
               Print("Socket connection failed: ", GetLastError());
            SocketClose(log_socket);
            log_socket = INVALID_HANDLE;
         }
         else if(EnableDebugLogging)
         {
            Print("Socket connected to ", LogSocketHost, ":", LogSocketPort);
         }
      }
      else if(EnableDebugLogging)
      {
         Print("Socket creation failed: ", GetLastError());
      }
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
   if(log_socket!=INVALID_HANDLE)
   {
      if(EnableDebugLogging)
         Print("Closing log socket");
      SocketClose(log_socket);
      log_socket = INVALID_HANDLE;
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

   if(entry==DEAL_ENTRY_IN || entry==DEAL_ENTRY_INOUT)
   {
      LogTrade("OPEN", ticket, magic, "mt4", symbol, order_type,
               lots, price, sl, tp, 0.0, remaining, now, comment);
      if(!IsTracked(ticket))
         AddTicket(ticket);
   }
   else if(entry==DEAL_ENTRY_OUT || entry==DEAL_ENTRY_OUT_BY)
   {
      LogTrade("CLOSE", ticket, magic, "mt4", symbol, order_type,
               lots, price, sl, tp, profit, remaining, now, comment);
      if(IsTracked(ticket) && remaining==0.0)
         RemoveTicket(ticket);
   }
}

void OnTick()
{
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
         LogTrade("OPEN", ticket, OrderMagicNumber(), "mt4", OrderSymbol(), OrderType(),
                  OrderLots(), OrderOpenPrice(), OrderStopLoss(), OrderTakeProfit(),
                  0.0, OrderLots(), now, OrderComment());
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
            LogTrade("CLOSE", ticket, OrderMagicNumber(), "mt4", OrderSymbol(),
                     OrderType(), OrderLots(), OrderClosePrice(), OrderStopLoss(),
                     OrderTakeProfit(), OrderProfit()+OrderSwap()+OrderCommission(),
                     0.0, now, OrderComment());
         }
         RemoveTicket(ticket);
         t--; // adjust index after removal
      }
   }
}

void OnTimer()
{
   FlushTradeBuffer();
   datetime now = UseBrokerTime ? TimeCurrent() : TimeLocal();

   if(EnableSocketLogging && log_socket==INVALID_HANDLE)
   {
      if(now - last_socket_attempt >= 60)
      {
         last_socket_attempt = now;
         log_socket = SocketCreate();
         if(log_socket!=INVALID_HANDLE)
         {
            if(!SocketConnect(log_socket, LogSocketHost, LogSocketPort, 1000))
            {
               if(EnableDebugLogging)
                  Print("Socket reconnection failed: ", GetLastError());
               SocketClose(log_socket);
               log_socket = INVALID_HANDLE;
            }
            else if(EnableDebugLogging)
            {
               Print("Socket reconnected to ", LogSocketHost, ":", LogSocketPort);
            }
         }
         else if(EnableDebugLogging)
         {
            Print("Socket creation failed: ", GetLastError());
         }
      }
   }
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

void LogTrade(string action, int ticket, int magic, string source,
              string symbol, int order_type, double lots, double price,
              double sl, double tp, double profit, double remaining,
              datetime time_event, string comment)
{
   if(trade_log_handle==INVALID_HANDLE)
      return;
   int id = NextEventId++;
   string line = StringFormat("%d;%s;%s;%s;%s;%d;%d;%s;%s;%d;%.2f;%.5f;%.5f;%.5f;%.2f;%s;%.2f",
      id,
      TimeToString(time_event, TIME_DATE|TIME_SECONDS),
      TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
      TimeToString(TimeLocal(), TIME_DATE|TIME_SECONDS),
      action, ticket, magic, source, symbol, order_type, lots, price, sl, tp, profit, comment, remaining);
   if(EnableDebugLogging)
   {
      FileSeek(trade_log_handle, 0, SEEK_END);
      FileWrite(trade_log_handle, line);
      FileFlush(trade_log_handle);
   }
   else
   {
      int n = ArraySize(trade_log_buffer);
      ArrayResize(trade_log_buffer, n+1);
      trade_log_buffer[n] = line;
      if(ArraySize(trade_log_buffer) >= LogBufferSize)
         FlushTradeBuffer();
   }

   string json = StringFormat("{\"event_id\":%d,\"event_time\":\"%s\",\"broker_time\":\"%s\",\"local_time\":\"%s\",\"action\":\"%s\",\"ticket\":%d,\"magic\":%d,\"source\":\"%s\",\"symbol\":\"%s\",\"order_type\":%d,\"lots\":%.2f,\"price\":%.5f,\"sl\":%.5f,\"tp\":%.5f,\"profit\":%.2f,\"comment\":\"%s\",\"remaining_lots\":%.2f}",
      id,
      EscapeJson(TimeToString(time_event, TIME_DATE|TIME_SECONDS)),
      EscapeJson(TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS)),
      EscapeJson(TimeToString(TimeLocal(), TIME_DATE|TIME_SECONDS)),
      EscapeJson(action), ticket, magic, EscapeJson(source), EscapeJson(symbol), order_type,
      lots, price, sl, tp, profit, EscapeJson(comment), remaining);

   if(log_socket!=INVALID_HANDLE)
   {
      uchar bytes[];
      StringToCharArray(json+"\n", bytes);
      if(SocketSend(log_socket, bytes, ArraySize(bytes)-1)==-1)
      {
         SocketClose(log_socket);
         log_socket = INVALID_HANDLE;
      }
   }
}

void ExportLogs(datetime ts)
{
   string src = LogDirectoryName + "\\trades_raw.csv";
   if(!FileIsExist(src))
      return;
   string dest = LogDirectoryName + "\\trades_" + TimeToString(ts, TIME_DATE|TIME_MINUTES) + ".csv";
   int in_h = FileOpen(src, FILE_CSV|FILE_READ|FILE_TXT|FILE_SHARE_READ|FILE_SHARE_WRITE, ';');
   if(in_h==INVALID_HANDLE)
      return;
   int out_h = FileOpen(dest, FILE_CSV|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE, ';');
   if(out_h!=INVALID_HANDLE)
   {
      while(!FileIsEnding(in_h))
      {
         string line = FileReadString(in_h);
         if(StringLen(line)>0)
            FileWrite(out_h, line);
      }
      FileClose(out_h);
   }
  FileClose(in_h);
 FileDelete(src);
}

void WriteMetrics(datetime ts)
{
   string fname = LogDirectoryName + "\\metrics.csv";
   int h = FileOpen(fname, FILE_CSV|FILE_READ|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE|FILE_SHARE_READ, ';');
   if(h==INVALID_HANDLE)
   {
      h = FileOpen(fname, FILE_CSV|FILE_WRITE|FILE_TXT|FILE_SHARE_WRITE, ';');
      if(h==INVALID_HANDLE)
         return;
      FileWrite(h, "time;magic;win_rate;avg_profit;trade_count;drawdown;sharpe");
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
      double profit_total = 0.0;
      double sum_profit = 0.0;
      double sum_sq_profit = 0.0;
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
      double variance = trades>0 ? sum_sq_profit/trades - MathPow(sum_profit/trades, 2) : 0.0;
      double stddev = variance>0 ? MathSqrt(variance) : 0.0;
      double sharpe = stddev>0 ? (sum_profit/trades)/stddev : 0.0;

      string line = StringFormat("%s;%d;%.3f;%.2f;%d;%.2f;%.3f", TimeToString(ts, TIME_DATE|TIME_MINUTES), magic, win_rate, avg_profit, trades, max_dd, sharpe);
      FileWrite(h, line);

      if(log_socket!=INVALID_HANDLE)
      {
         string json = StringFormat("{\"type\":\"metrics\",\"time\":\"%s\",\"magic\":%d,\"win_rate\":%.3f,\"avg_profit\":%.2f,\"trade_count\":%d,\"drawdown\":%.2f,\"sharpe\":%.3f}",
            EscapeJson(TimeToString(ts, TIME_DATE|TIME_MINUTES)), magic, win_rate, avg_profit, trades, max_dd, sharpe);
         uchar bytes[];
         StringToCharArray(json+"\n", bytes);
         if(SocketSend(log_socket, bytes, ArraySize(bytes)-1)==-1)
         {
            SocketClose(log_socket);
            log_socket = INVALID_HANDLE;
         }
      }
   }

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
      FileWrite(h, "time;model_id;hit_rate;profit_factor");
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
      FileWrite(h, line);
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
     FileWrite(out_h, lines[i]);
  FileClose(out_h);
}

void FlushTradeBuffer()
{
   if(trade_log_handle==INVALID_HANDLE)
      return;
   int n = ArraySize(trade_log_buffer);
   if(n==0)
      return;
   FileSeek(trade_log_handle, 0, SEEK_END);
   for(int i=0; i<n; i++)
      FileWrite(trade_log_handle, trade_log_buffer[i]);
   FileFlush(trade_log_handle);
   ArrayResize(trade_log_buffer, 0);
}
