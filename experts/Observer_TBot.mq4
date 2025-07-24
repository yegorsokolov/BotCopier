#property strict
#include "model_interface.mqh"

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

int timer_handle;

int      tracked_tickets[];
int      target_magics[];
string   track_symbols[];
datetime last_export = 0;

int OnInit()
{
   EventSetTimer(1);
   ArrayResize(tracked_tickets, 0);

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

   last_export = UseBrokerTime ? TimeCurrent() : TimeLocal();
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
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
   for(int i=0; i<ArraySize(tracked_tickets); i++)
      if(tracked_tickets[i]==ticket)
         return(true);
   return(false);
}

void AddTicket(int ticket)
{
   int n = ArraySize(tracked_tickets);
   ArrayResize(tracked_tickets, n+1);
   tracked_tickets[n] = ticket;
}

void RemoveTicket(int ticket)
{
   for(int i=0; i<ArraySize(tracked_tickets); i++)
   {
      if(tracked_tickets[i]==ticket)
      {
         for(int j=i; j<ArraySize(tracked_tickets)-1; j++)
            tracked_tickets[j] = tracked_tickets[j+1];
         ArrayResize(tracked_tickets, ArraySize(tracked_tickets)-1);
         break;
      }
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
                  0.0, now, OrderComment());
         AddTicket(ticket);
      }
   }
   ArrayResize(current, cur_idx);

   for(int t=0; t<ArraySize(tracked_tickets); t++)
   {
      int ticket = tracked_tickets[t];
      bool still_open = false;
      for(int c=0; c<cur_idx; c++)
      {
         if(current[c]==ticket)
         {
            still_open = true;
            break;
         }
      }
      if(!still_open)
      {
         if(OrderSelect(ticket, SELECT_BY_TICKET, MODE_HISTORY))
         {
            LogTrade("CLOSE", ticket, OrderMagicNumber(), "mt4", OrderSymbol(),
                     OrderType(), OrderLots(), OrderClosePrice(), OrderStopLoss(),
                     OrderTakeProfit(), OrderProfit()+OrderSwap()+OrderCommission(),
                     now, OrderComment());
         }
         RemoveTicket(ticket);
         t--; // adjust index after removal
      }
   }
}

void OnTimer()
{
   datetime now = UseBrokerTime ? TimeCurrent() : TimeLocal();
   if(now - last_export < LearningExportIntervalMinutes*60)
      return;

   ExportLogs(now);
   ManageMetrics(now);
   last_export = now;
}

void LogTrade(string action, int ticket, int magic, string source,
              string symbol, int order_type, double lots, double price,
              double sl, double tp, double profit, datetime time_event, string comment)
{
   string fname = LogDirectoryName + "\\trades_raw.csv";
   int f = FileOpen(fname, FILE_CSV|FILE_WRITE|FILE_READ|FILE_TXT|FILE_SHARE_WRITE, ';');
   if(f==INVALID_HANDLE)
      return;
   bool need_header = (FileSize(f)==0);
   FileSeek(f, 0, SEEK_END);
   if(need_header)
   {
      string header = "event_time;broker_time;local_time;action;ticket;magic;source;symbol;order_type;lots;price;sl;tp;profit;comment";
      FileWrite(f, header);
   }
   string line = StringFormat("%s;%s;%s;%s;%d;%d;%s;%s;%d;%.2f;%.5f;%.5f;%.5f;%.2f;%s",
      TimeToString(time_event, TIME_DATE|TIME_SECONDS),
      TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS),
      TimeToString(TimeLocal(), TIME_DATE|TIME_SECONDS),
      action, ticket, magic, source, symbol, order_type, lots, price, sl, tp, profit, comment);
   FileWrite(f, line);
   FileClose(f);
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
