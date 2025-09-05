#property strict

// Strategy template for generated expert advisor.

// Seconds between model reload checks for session switching.  A value of 0 disables the timer.
extern int ReloadModelInterval = 60;

double g_coeffs[];
double g_threshold;
double g_feature_mean[];
double g_feature_std[];

// __SESSION_MODELS__

void SelectSessionModel()
{
    int h = TimeHour(TimeCurrent());
    if(h < 8)
    {
        ArrayCopy(g_coeffs, g_coeffs_asian);
        g_threshold = g_threshold_asian;
        ArrayCopy(g_feature_mean, g_feature_mean_asian);
        ArrayCopy(g_feature_std, g_feature_std_asian);
    }
    else if(h < 16)
    {
        ArrayCopy(g_coeffs, g_coeffs_london);
        g_threshold = g_threshold_london;
        ArrayCopy(g_feature_mean, g_feature_mean_london);
        ArrayCopy(g_feature_std, g_feature_std_london);
    }
    else
    {
        ArrayCopy(g_coeffs, g_coeffs_newyork);
        g_threshold = g_threshold_newyork;
        ArrayCopy(g_feature_mean, g_feature_mean_newyork);
        ArrayCopy(g_feature_std, g_feature_std_newyork);
    }
}

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

int OnInit()
{
    LoadWal(TRADE_WAL, g_trade_queue);
    LoadWal(METRIC_WAL, g_metric_queue);
    SelectSessionModel();
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
    SelectSessionModel();
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

double ScoreModel()
{
    double z = g_coeffs[0];
    for(int i = 1; i < ArraySize(g_coeffs); i++)
    {
        double val = GetFeature(i - 1);
        double norm = (val - g_feature_mean[i - 1]) / g_feature_std[i - 1];
        z += g_coeffs[i] * norm;
    }
    return 1.0 / (1.0 + MathExp(-z));
}
