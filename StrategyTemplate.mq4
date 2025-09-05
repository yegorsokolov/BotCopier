#property strict

// Strategy template for generated expert advisor.
//
// The expert loads model parameters from a comma separated file located in the
// ``Files`` directory of the terminal.  The file ``model_params.csv`` is
// expected to contain two lines: the first with coefficient values and the
// second with decision thresholds.  A timer periodically checks for updates and
// reloads the values without requiring recompilation.

// Seconds between model reload checks.  A value of 0 disables the timer.
extern int ReloadModelInterval = 60;

double g_coeffs[];
double g_thresholds[];
datetime g_last_params_time = 0;

string g_trade_queue[];
string g_metric_queue[];
string TRADE_WAL = "trades.wal";
string METRIC_WAL = "metrics.wal";

void LoadWal(string file, string &queue[])
{
    int handle = FileOpen(file, FILE_READ|FILE_ANSI|FILE_TXT);
    if(handle == INVALID_HANDLE)
        return;
    while(!FileIsEnding(handle))
    {
        string line = FileReadString(handle);
        if(StringLen(line) > 0)
        {
            int n = ArraySize(queue);
            ArrayResize(queue, n + 1);
            queue[n] = line;
        }
    }
    FileClose(handle);
}

void AppendWal(string file, string payload)
{
    int handle = FileOpen(file, FILE_READ|FILE_WRITE|FILE_ANSI|FILE_TXT);
    if(handle == INVALID_HANDLE)
        return;
    FileSeek(handle, 0, SEEK_END);
    FileWrite(handle, payload);
    FileClose(handle);
}

void QueueTrade(string payload)
{
    int n = ArraySize(g_trade_queue);
    ArrayResize(g_trade_queue, n + 1);
    g_trade_queue[n] = payload;
    AppendWal(TRADE_WAL, payload);
}

void QueueMetric(string payload)
{
    int n = ArraySize(g_metric_queue);
    ArrayResize(g_metric_queue, n + 1);
    g_metric_queue[n] = payload;
    AppendWal(METRIC_WAL, payload);
}

bool SendTrade(string payload)
{
    return(true); // placeholder for transmission
}

bool SendMetric(string payload)
{
    return(true); // placeholder for transmission
}

void FlushTrades()
{
    int i = 0;
    while(i < ArraySize(g_trade_queue))
    {
        if(SendTrade(g_trade_queue[i]))
        {
            for(int j = i; j < ArraySize(g_trade_queue) - 1; j++)
                g_trade_queue[j] = g_trade_queue[j+1];
            ArrayResize(g_trade_queue, ArraySize(g_trade_queue) - 1);
        }
        else
            i++;
    }
    int handle = FileOpen(TRADE_WAL, FILE_WRITE|FILE_ANSI|FILE_TXT);
    if(handle != INVALID_HANDLE)
    {
        for(i = 0; i < ArraySize(g_trade_queue); i++)
            FileWrite(handle, g_trade_queue[i]);
        FileClose(handle);
    }
}

void FlushMetrics()
{
    int i = 0;
    while(i < ArraySize(g_metric_queue))
    {
        if(SendMetric(g_metric_queue[i]))
        {
            for(int j = i; j < ArraySize(g_metric_queue) - 1; j++)
                g_metric_queue[j] = g_metric_queue[j+1];
            ArrayResize(g_metric_queue, ArraySize(g_metric_queue) - 1);
        }
        else
            i++;
    }
    int handle = FileOpen(METRIC_WAL, FILE_WRITE|FILE_ANSI|FILE_TXT);
    if(handle != INVALID_HANDLE)
    {
        for(i = 0; i < ArraySize(g_metric_queue); i++)
            FileWrite(handle, g_metric_queue[i]);
        FileClose(handle);
    }
}

// Read coefficients and thresholds from the Files\model_params.csv file.
bool LoadParameters()
{
    int handle = FileOpen("model_params.csv", FILE_READ|FILE_ANSI);
    if(handle == INVALID_HANDLE)
    {
        Print("Failed to open model_params.csv: ", GetLastError());
        return false;
    }

    string line;
    if(!FileIsEnding(handle))
    {
        line = FileReadString(handle);
        string parts[];
        int n = StringSplit(line, ",", parts);
        ArrayResize(g_coeffs, n);
        for(int i = 0; i < n; i++)
            g_coeffs[i] = StrToDouble(parts[i]);
    }

    if(!FileIsEnding(handle))
    {
        line = FileReadString(handle);
        string parts[];
        int n = StringSplit(line, ",", parts);
        ArrayResize(g_thresholds, n);
        for(int i = 0; i < n; i++)
            g_thresholds[i] = StrToDouble(parts[i]);
    }

    g_last_params_time = (datetime)FileGetInteger(handle, FILE_MODIFY_DATE);
    FileClose(handle);
    Print("Loaded ", ArraySize(g_coeffs), " coeffs and ", ArraySize(g_thresholds), " thresholds");
    return true;
}

int OnInit()
{
    LoadWal(TRADE_WAL, g_trade_queue);
    LoadWal(METRIC_WAL, g_metric_queue);
    if(!LoadParameters())
        return(INIT_FAILED);
    if(ReloadModelInterval > 0)
        EventSetTimer(ReloadModelInterval);
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
    if(ReloadModelInterval > 0)
        EventKillTimer();
}

void OnTimer()
{
    FlushTrades();
    FlushMetrics();
    int handle = FileOpen("model_params.csv", FILE_READ|FILE_ANSI);
    if(handle == INVALID_HANDLE)
        return;
    datetime mod = (datetime)FileGetInteger(handle, FILE_MODIFY_DATE);
    FileClose(handle);
    if(mod > g_last_params_time)
        LoadParameters();
}

double GetFeature(int idx)
{
    switch(idx)
    {
    case 0: return MarketInfo(Symbol(), MODE_SPREAD); // spread
    case 1: return TimeHour(TimeCurrent()); // hour
    }
    return 0.0;
}
