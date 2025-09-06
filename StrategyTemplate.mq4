#property strict

// Strategy template for generated expert advisor.

// Seconds between model reload checks for session switching.  A value of 0 disables the timer.
extern int ReloadModelInterval = 60;

double g_coeffs[];
double g_threshold;
double g_feature_mean[];
double g_feature_std[];

double g_coeffs_asian[] = {0.0, 0.0};
double g_threshold_asian = 0.5;
double g_feature_mean_asian[] = {};
double g_feature_std_asian[] = {};
double g_coeffs_london[] = {0.0, 0.0};
double g_threshold_london = 0.5;
double g_feature_mean_london[] = {};
double g_feature_std_london[] = {};
double g_coeffs_newyork[] = {0.0, 0.0};
double g_threshold_newyork = 0.5;
double g_feature_mean_newyork[] = {};
double g_feature_std_newyork[] = {};

datetime g_last_model_reload = 0;

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
int g_trade_head = 0;
int g_trade_tail = 0;
int g_metric_head = 0;
int g_metric_tail = 0;
string TRADE_WAL = "trades.wal";
string METRIC_WAL = "metrics.wal";

// Remote endpoints for low-latency transport
string TRADE_HOST = "127.0.0.1";
int    TRADE_PORT = 50052;
string METRIC_HOST = "127.0.0.1";
int    METRIC_PORT = 50053;

// Counters for observability of send failures
int g_trade_send_failures = 0;
int g_metric_send_failures = 0;

string WalEncode(string payload)
{
    uchar src[], dst[];
    StringToCharArray(payload, src);
    CryptEncode(CRYPT_ARCHIVE_GZIP, src, dst);
    return(CharArrayToString(dst));
}

string WalDecode(string payload)
{
    uchar src[], dst[];
    StringToCharArray(payload, src);
    if(CryptDecode(CRYPT_ARCHIVE_GZIP, src, dst))
        return(CharArrayToString(dst));
    return(payload);
}

int GetWalSize(string file)
{
    int handle = FileOpen(file, FILE_READ|FILE_ANSI|FILE_TXT);
    if(handle == INVALID_HANDLE)
        return(0);
    int size = FileSize(handle);
    FileClose(handle);
    return(size);
}

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
            string decoded = WalDecode(line);
            int n = ArraySize(queue);
            ArrayResize(queue, n + 1);
            queue[n] = decoded;
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
    FileWrite(handle, WalEncode(payload));
    FileClose(handle);
}

void QueueTrade(string payload)
{
    if(g_trade_tail >= ArraySize(g_trade_queue))
        ArrayResize(g_trade_queue, g_trade_tail + 1);
    g_trade_queue[g_trade_tail] = payload;
    g_trade_tail++;
    AppendWal(TRADE_WAL, payload);
}

void QueueMetric(string payload)
{
    if(g_metric_tail >= ArraySize(g_metric_queue))
        ArrayResize(g_metric_queue, g_metric_tail + 1);
    g_metric_queue[g_metric_tail] = payload;
    g_metric_tail++;
    AppendWal(METRIC_WAL, payload);
}

void ReportWalSizes()
{
    QueueMetric("trade_wal_size=" + IntegerToString(GetWalSize(TRADE_WAL)));
    QueueMetric("metric_wal_size=" + IntegerToString(GetWalSize(METRIC_WAL)));
}

bool SendTrade(string payload)
{
    uchar data[];
    int len = StringToCharArray(payload, data);
    int sock = SocketCreate();
    if(sock == INVALID_HANDLE || !SocketConnect(sock, TRADE_HOST, TRADE_PORT))
    {
        if(sock != INVALID_HANDLE)
            SocketClose(sock);
        g_trade_send_failures++;
        QueueMetric("trade_send_failures=" + IntegerToString(g_trade_send_failures));
        return(false);
    }

    int sent = SocketSend(sock, data, len);
    SocketClose(sock);
    if(sent != len)
    {
        g_trade_send_failures++;
        QueueMetric("trade_send_failures=" + IntegerToString(g_trade_send_failures));
        return(false);
    }
    return(true);
}

bool SendMetric(string payload)
{
    uchar data[];
    int len = StringToCharArray(payload, data);
    int sock = SocketCreate();
    if(sock == INVALID_HANDLE || !SocketConnect(sock, METRIC_HOST, METRIC_PORT))
    {
        if(sock != INVALID_HANDLE)
            SocketClose(sock);
        g_metric_send_failures++;
        QueueMetric("metric_send_failures=" + IntegerToString(g_metric_send_failures));
        return(false);
    }

    int sent = SocketSend(sock, data, len);
    SocketClose(sock);
    if(sent != len)
    {
        g_metric_send_failures++;
        QueueMetric("metric_send_failures=" + IntegerToString(g_metric_send_failures));
        return(false);
    }
    return(true);
}

void FlushTrades()
{
    while(g_trade_head < g_trade_tail)
    {
        string payload = WalDecode(g_trade_queue[g_trade_head]);
        if(!SendTrade(payload))
            break;
        g_trade_head++;
    }

    int handle = FileOpen(TRADE_WAL, FILE_WRITE|FILE_ANSI|FILE_TXT);
    if(handle != INVALID_HANDLE)
    {
        for(int i = g_trade_head; i < g_trade_tail; i++)
            FileWrite(handle, WalEncode(g_trade_queue[i]));
        FileClose(handle);
    }

    if(g_trade_head > 0)
    {
        if(g_trade_head == g_trade_tail)
        {
            ArrayResize(g_trade_queue, 0);
            g_trade_head = 0;
            g_trade_tail = 0;
        }
        else
        {
            int remaining = g_trade_tail - g_trade_head;
            for(int i = 0; i < remaining; i++)
                g_trade_queue[i] = g_trade_queue[g_trade_head + i];
            ArrayResize(g_trade_queue, remaining);
            g_trade_head = 0;
            g_trade_tail = remaining;
        }
    }
}

void FlushMetrics()
{
    while(g_metric_head < g_metric_tail)
    {
        string payload = WalDecode(g_metric_queue[g_metric_head]);
        if(!SendMetric(payload))
            break;
        g_metric_head++;
    }

    int handle = FileOpen(METRIC_WAL, FILE_WRITE|FILE_ANSI|FILE_TXT);
    if(handle != INVALID_HANDLE)
    {
        for(int i = g_metric_head; i < g_metric_tail; i++)
            FileWrite(handle, WalEncode(g_metric_queue[i]));
        FileClose(handle);
    }

    if(g_metric_head > 0)
    {
        if(g_metric_head == g_metric_tail)
        {
            ArrayResize(g_metric_queue, 0);
            g_metric_head = 0;
            g_metric_tail = 0;
        }
        else
        {
            int remaining = g_metric_tail - g_metric_head;
            for(int i = 0; i < remaining; i++)
                g_metric_queue[i] = g_metric_queue[g_metric_head + i];
            ArrayResize(g_metric_queue, remaining);
            g_metric_head = 0;
            g_metric_tail = remaining;
        }
    }
}

int OnInit()
{
    LoadWal(TRADE_WAL, g_trade_queue);
    g_trade_head = 0;
    g_trade_tail = ArraySize(g_trade_queue);
    LoadWal(METRIC_WAL, g_metric_queue);
    g_metric_head = 0;
    g_metric_tail = ArraySize(g_metric_queue);
    SelectSessionModel();
    g_last_model_reload = TimeCurrent();
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
    ReportWalSizes();
    SelectSessionModel();
    g_last_model_reload = TimeCurrent();
}

double GetFeature(int idx)
{
    switch(idx)
    {
    case 0: return MarketInfo(Symbol(), MODE_SPREAD); // spread
    case 1: return TimeHour(TimeCurrent()); // hour
    case 2: return iVolume(Symbol(), PERIOD_CURRENT, 0); // volume
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

void OnTick()
{
    datetime now = TimeCurrent();
    if(ReloadModelInterval > 0 && now - g_last_model_reload >= ReloadModelInterval)
    {
        SelectSessionModel();
        g_last_model_reload = now;
    }

    double prob = ScoreModel();
    string decision = "hold";
    if(prob > g_threshold)
    {
        OrderSend(Symbol(), OP_BUY, 0.1, Ask, 3, 0, 0);
        decision = "buy";
    }
    else if((1.0 - prob) > g_threshold)
    {
        OrderSend(Symbol(), OP_SELL, 0.1, Bid, 3, 0, 0);
        decision = "sell";
    }
    QueueTrade("decision=" + decision + ",prob=" + DoubleToString(prob, 8));
}
