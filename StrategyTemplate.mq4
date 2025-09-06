#property strict

// Strategy template for generated expert advisor.

// Seconds between model reload checks for session switching.  A value of 0 disables the timer.
extern int ReloadModelInterval = 60;

double g_coeffs[];
double g_threshold;
double g_feature_mean[];
double g_feature_std[];
double g_lot_coeffs[];
double g_sl_coeffs[];
double g_tp_coeffs[];
double g_conformal_lower;
double g_conformal_upper;

double g_coeffs_asian[] = {0.0, 0.0};
double g_threshold_asian = 0.5;
double g_feature_mean_asian[] = {};
double g_feature_std_asian[] = {};
double g_lot_coeffs_asian[] = {0.0, 0.0};
double g_sl_coeffs_asian[] = {0.0, 0.0};
double g_tp_coeffs_asian[] = {0.0, 0.0};
double g_conformal_lower_asian = 0.0;
double g_conformal_upper_asian = 1.0;
double g_coeffs_london[] = {0.0, 0.0};
double g_threshold_london = 0.5;
double g_feature_mean_london[] = {};
double g_feature_std_london[] = {};
double g_lot_coeffs_london[] = {0.0, 0.0};
double g_sl_coeffs_london[] = {0.0, 0.0};
double g_tp_coeffs_london[] = {0.0, 0.0};
double g_conformal_lower_london = 0.0;
double g_conformal_upper_london = 1.0;
double g_coeffs_newyork[] = {0.0, 0.0};
double g_threshold_newyork = 0.5;
double g_feature_mean_newyork[] = {};
double g_feature_std_newyork[] = {};
double g_lot_coeffs_newyork[] = {0.0, 0.0};
double g_sl_coeffs_newyork[] = {0.0, 0.0};
double g_tp_coeffs_newyork[] = {0.0, 0.0};
double g_conformal_lower_newyork = 0.0;
double g_conformal_upper_newyork = 1.0;

datetime g_last_model_reload = 0;

void SelectSessionModel()
{
    int h = TimeHour(TimeCurrent());
    if(h < 8)
    {
        ArrayCopy(g_coeffs, g_coeffs_asian);
        ArrayCopy(g_lot_coeffs, g_lot_coeffs_asian);
        ArrayCopy(g_sl_coeffs, g_sl_coeffs_asian);
        ArrayCopy(g_tp_coeffs, g_tp_coeffs_asian);
        g_threshold = g_threshold_asian;
        ArrayCopy(g_feature_mean, g_feature_mean_asian);
        ArrayCopy(g_feature_std, g_feature_std_asian);
        g_conformal_lower = g_conformal_lower_asian;
        g_conformal_upper = g_conformal_upper_asian;
    }
    else if(h < 16)
    {
        ArrayCopy(g_coeffs, g_coeffs_london);
        ArrayCopy(g_lot_coeffs, g_lot_coeffs_london);
        ArrayCopy(g_sl_coeffs, g_sl_coeffs_london);
        ArrayCopy(g_tp_coeffs, g_tp_coeffs_london);
        g_threshold = g_threshold_london;
        ArrayCopy(g_feature_mean, g_feature_mean_london);
        ArrayCopy(g_feature_std, g_feature_std_london);
        g_conformal_lower = g_conformal_lower_london;
        g_conformal_upper = g_conformal_upper_london;
    }
    else
    {
        ArrayCopy(g_coeffs, g_coeffs_newyork);
        ArrayCopy(g_lot_coeffs, g_lot_coeffs_newyork);
        ArrayCopy(g_sl_coeffs, g_sl_coeffs_newyork);
        ArrayCopy(g_tp_coeffs, g_tp_coeffs_newyork);
        g_threshold = g_threshold_newyork;
        ArrayCopy(g_feature_mean, g_feature_mean_newyork);
        ArrayCopy(g_feature_std, g_feature_std_newyork);
        g_conformal_lower = g_conformal_lower_newyork;
        g_conformal_upper = g_conformal_upper_newyork;
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

int g_decision_id = 0;

#import "grpc_client.dll"
void   grpc_client_init(string trade_host, int trade_port, string metric_host, int metric_port);
void   grpc_enqueue_trade(string payload);
void   grpc_enqueue_metric(string payload);
void   grpc_flush();
#import

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
    grpc_enqueue_trade(payload);
    return(true);
}

bool SendMetric(string payload)
{
    grpc_enqueue_metric(payload);
    return(true);
}

void FlushTrades()
{
    while(g_trade_head < g_trade_tail)
    {
        string payload = WalDecode(g_trade_queue[g_trade_head]);
        SendTrade(payload);
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
        SendMetric(payload);
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
    grpc_client_init(TRADE_HOST, TRADE_PORT, METRIC_HOST, METRIC_PORT);
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
    grpc_flush();
    ReportWalSizes();
    SelectSessionModel();
    g_last_model_reload = TimeCurrent();
}

double RollingCorrelation(string sym1, string sym2, int window)
{
    double sumx = 0, sumy = 0, sumxy = 0, sumx2 = 0, sumy2 = 0;
    for(int i = 0; i < window; i++)
    {
        double x = iClose(sym1, PERIOD_CURRENT, i);
        double y = iClose(sym2, PERIOD_CURRENT, i);
        sumx += x;
        sumy += y;
        sumxy += x * y;
        sumx2 += x * x;
        sumy2 += y * y;
    }
    double num = window * sumxy - sumx * sumy;
    double den = MathSqrt(window * sumx2 - sumx * sumx) * MathSqrt(window * sumy2 - sumy * sumy);
    if(den == 0)
        return(0);
    return(num / den);
}

double GetFeature(int idx)
{
    switch(idx)
    {
    case 0: return MarketInfo(Symbol(), MODE_SPREAD); // spread
    case 1: return OrderSlippage(); // slippage
    case 2: return AccountEquity(); // equity
    case 3: return AccountMarginLevel(); // margin_level
    case 4: return iVolume(Symbol(), PERIOD_CURRENT, 0); // volume
    case 5: return MathSin(TimeHour(TimeCurrent())*2*MathPi()/24); // hour_sin
    case 6: return MathCos(TimeHour(TimeCurrent())*2*MathPi()/24); // hour_cos
    }
    return 0.0;
}


double ApplyModel(double &coeffs[])
{
    double z = coeffs[0];
    for(int i = 1; i < ArraySize(coeffs); i++)
    {
        double val = GetFeature(i - 1);
        double norm = (val - g_feature_mean[i - 1]) / g_feature_std[i - 1];
        z += coeffs[i] * norm;
    }
    return z;
}

double ScoreModel()
{
    return 1.0 / (1.0 + MathExp(-ApplyModel(g_coeffs)));
}

double PredictLot()
{
    return MathMax(0.01, ApplyModel(g_lot_coeffs));
}

double PredictSLDistance()
{
    return MathAbs(ApplyModel(g_sl_coeffs));
}

double PredictTPDistance()
{
    return MathAbs(ApplyModel(g_tp_coeffs));
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
    double lot = PredictLot();
    double sl = PredictSLDistance();
    double tp = PredictTPDistance();
    string decision = "hold";
    bool uncertain = (prob >= g_conformal_lower && prob <= g_conformal_upper);
    if(uncertain)
    {
        decision = "skip";
    }
    else if(prob > g_threshold)
    {
        double sl_price = Ask - sl * Point;
        double tp_price = Ask + tp * Point;
        OrderSend(Symbol(), OP_BUY, lot, Ask, 3, sl_price, tp_price);
        decision = "buy";
    }
    else if((1.0 - prob) > g_threshold)
    {
        double sl_price = Bid + sl * Point;
        double tp_price = Bid - tp * Point;
        OrderSend(Symbol(), OP_SELL, lot, Bid, 3, sl_price, tp_price);
        decision = "sell";
    }
    string features = "";
    int feat_count = ArraySize(g_feature_mean);
    for(int i = 0; i < feat_count; i++)
    {
        if(i > 0)
            features += ":";
        features += DoubleToString(GetFeature(i), 8);
    }
    g_decision_id++;
    QueueTrade(
        "decision_id=" + IntegerToString(g_decision_id) +
        ",decision=" + decision +
        ",prob=" + DoubleToString(prob, 8) +
        ",lot=" + DoubleToString(lot, 2) +
        ",sl=" + DoubleToString(sl, 2) +
        ",tp=" + DoubleToString(tp, 2) +
        ",features=" + features
    );
}
