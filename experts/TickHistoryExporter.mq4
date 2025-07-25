#property strict
#property script_show_inputs

input string OutDir = "observer_logs"; // Directory inside MQL4/Files
input string Symbols = "";            // Optional comma separated symbol list

struct SymbolRange
{
   string   symbol;
   datetime start;
   datetime end;
};

int OnStart()
{
   HistorySelect(0, TimeCurrent());
   int deals = HistoryDealsTotal();
   SymbolRange ranges[];
   string filter[];
   int filter_count = 0;
   if(StringLen(Symbols) > 0)
      filter_count = StringSplit(Symbols, ',', filter);

   for(int i=0; i<deals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(!HistoryDealSelect(ticket))
         continue;
      string sym = HistoryDealGetString(ticket, DEAL_SYMBOL);
      datetime tm = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      bool match = (filter_count == 0);
      if(!match)
      {
         for(int j=0; j<filter_count; j++)
            if(StringCompare(filter[j], sym, true)==0) { match=true; break; }
      }
      if(!match) continue;
      int idx = -1;
      for(int r=0; r<ArraySize(ranges); r++)
         if(StringCompare(ranges[r].symbol, sym, true)==0) { idx=r; break; }
      if(idx < 0)
      {
         SymbolRange sr;
         sr.symbol = sym;
         sr.start = tm;
         sr.end = tm;
         int n = ArraySize(ranges);
         ArrayResize(ranges, n+1);
         ranges[n] = sr;
      }
      else
      {
         if(tm < ranges[idx].start) ranges[idx].start = tm;
         if(tm > ranges[idx].end)   ranges[idx].end   = tm;
      }
   }

   for(int i=0; i<ArraySize(ranges); i++)
      ExportTicks(ranges[i].symbol, ranges[i].start, ranges[i].end);

   return(INIT_SUCCEEDED);
}

void ExportTicks(string symbol, datetime from, datetime to)
{
   MqlTick ticks[];
   string fname = StringFormat("%s/ticks_%s.csv", OutDir, symbol);
   int fh = FileOpen(fname, FILE_WRITE|FILE_CSV|FILE_TXT|FILE_SHARE_WRITE, ';');
   if(fh==INVALID_HANDLE)
   {
      Print("Failed to open ", fname, " : ", GetLastError());
      return;
   }
   FileWrite(fh, "time;bid;ask;last;volume");
   datetime cur = from;
   long total = 0;
   while(cur <= to)
   {
      int copied = CopyTicksRange(symbol, ticks, cur, to, COPY_TICKS_ALL);
      if(copied <= 0)
         break;
      for(int i=0; i<copied; i++)
      {
         FileWrite(fh,
                   TimeToString(ticks[i].time, TIME_DATE|TIME_SECONDS),
                   DoubleToString(ticks[i].bid, _Digits),
                   DoubleToString(ticks[i].ask, _Digits),
                   DoubleToString(ticks[i].last, _Digits),
                   DoubleToString(ticks[i].volume, 2));
         cur = ticks[i].time + 1;
         total++;
      }
   }
   FileClose(fh);
   Print(StringFormat("Exported %d ticks for %s", total, symbol));
}
