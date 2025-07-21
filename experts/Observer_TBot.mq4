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
extern string LogDirectoryName              = "observer_logs";
extern bool   EnableDebugLogging            = false;
extern bool   UseBrokerTime                 = true;
extern string SymbolsToTrack                = ""; // empty=all

int timer_handle;

int OnInit()
{
   EventSetTimer(1);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
}

void OnTick()
{
   // placeholder for trade monitoring
}

void OnTimer()
{
   // placeholder for periodic export and panel update
}

void LogTrade(string action, int ticket, int magic, string source,
              string symbol, int order_type, double lots, double price,
              double sl, double tp, datetime time_event, string comment)
{
   string fname = LogDirectoryName + "\\trades_raw.csv";
   int f = FileOpen(fname, FILE_CSV|FILE_WRITE|FILE_READ|FILE_TXT|FILE_SHARE_WRITE, ';');
   if(f==INVALID_HANDLE)
      return;
   FileSeek(f, 0, SEEK_END);
   string line = StringFormat("%s;%d;%d;%s;%s;%d;%.2f;%.5f;%.5f;%.5f;%s",
      TimeToString(time_event, TIME_DATE|TIME_SECONDS),
      ticket, magic, source, symbol, order_type, lots, price, sl, tp, comment);
   FileWrite(f, line);
   FileClose(f);
}
