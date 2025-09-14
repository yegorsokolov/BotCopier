#property strict

// Strategy template for generated expert advisor.

double g_coeffs[];
double g_threshold;
double g_feature_mean[];
double g_feature_std[];
double g_lot_coeffs[];
double g_sl_coeffs[];
double g_tp_coeffs[];
double g_conformal_lower;
double g_conformal_upper;
int g_order_types[];
double g_order_thresholds[];
// __SESSION_MODELS__

bool g_use_transformer = false;
// __TRANSFORMER_PARAMS_START__
int g_transformer_window = 0;
int g_transformer_dim = 0;
int g_transformer_feat_dim = 0;
double g_tfeature_mean[1] = {0.0};
double g_tfeature_std[1] = {1.0};
double g_embed_weight[1] = {0.0};
double g_embed_bias[1] = {0.0};
double g_q_weight[1] = {0.0};
double g_q_bias[1] = {0.0};
double g_k_weight[1] = {0.0};
double g_k_bias[1] = {0.0};
double g_v_weight[1] = {0.0};
double g_v_bias[1] = {0.0};
double g_out_weight[1] = {0.0};
double g_out_bias = 0.0;
// __TRANSFORMER_PARAMS_END__

double g_transformer_buffer[];
int g_transformer_seq_len = 0;

// __SYMBOL_EMBEDDINGS_START__
double GraphEmbedding(int idx)
{
    return 0.0;
}
// __SYMBOL_EMBEDDINGS_END__

// __SYMBOL_THRESHOLDS_START__
double SymbolThreshold()
{
    return g_threshold;
}
// __SYMBOL_THRESHOLDS_END__

string g_calendar_file = "__CALENDAR_FILE__";
datetime g_cal_times[];
double g_cal_impacts[];
int g_cal_loaded = 0;
int g_cal_count = 0;
int g_calendar_window = 60;

void LoadCalendar()
{
    if(g_cal_loaded) return;
    int handle = FileOpen(g_calendar_file, FILE_READ|FILE_CSV);
    if(handle < 0)
    {
        g_cal_loaded = 1;
        return;
    }
    FileReadString(handle); // header
    while(!FileIsEnding(handle))
    {
        string sTime = FileReadString(handle);
        if(sTime == "") break;
        double impact = StrToDouble(FileReadString(handle));
        FileReadString(handle);
        ArrayResize(g_cal_times, g_cal_count+1);
        ArrayResize(g_cal_impacts, g_cal_count+1);
        g_cal_times[g_cal_count] = StringToTime(sTime);
        g_cal_impacts[g_cal_count] = impact;
        g_cal_count++;
    }
    FileClose(handle);
    g_cal_loaded = 1;
}

double CalendarFlag()
{
    LoadCalendar();
    datetime now = TimeCurrent();
    for(int i=0; i<g_cal_count; i++)
    {
        if(MathAbs((double)(now - g_cal_times[i])) <= g_calendar_window*60) return 1.0;
    }
    return 0.0;
}

double CalendarImpact()
{
    LoadCalendar();
    datetime now = TimeCurrent();
    double maxImp = 0.0;
    for(int i=0; i<g_cal_count; i++)
    {
        double diff = MathAbs((double)(now - g_cal_times[i]));
        if(diff <= g_calendar_window*60 && g_cal_impacts[i] > maxImp)
            maxImp = g_cal_impacts[i];
    }
    return maxImp;
}

// __INDICATOR_FUNCTIONS__

int GetOrderType(int idx)
{
    if(idx < ArraySize(g_order_types))
        return g_order_types[idx];
    return OP_BUY;
}

double OrderThreshold(int order_type)
{
    for(int i = 0; i < ArraySize(g_order_types); i++)
    {
        if(g_order_types[i] == order_type)
            return g_order_thresholds[i];
    }
    return SymbolThreshold();
}

int RouteModel()
{
    double features[2];
    features[0] = iStdDev(Symbol(), PERIOD_CURRENT, 14, 0, MODE_SMA, PRICE_CLOSE, 0);
    features[1] = TimeHour(TimeCurrent());
    int count = ArraySize(g_router_intercept);
    int fcount = ArraySize(g_router_feature_mean);
    double best = -1e10;
    int best_idx = 0;
    for(int i = 0; i < count; i++)
    {
        double z = g_router_intercept[i];
        for(int f = 0; f < fcount; f++)
        {
            double val = features[f];
            z += (val - g_router_feature_mean[f]) / g_router_feature_std[f] * g_router_coeffs[i*fcount + f];
        }
        if(z > best)
        {
            best = z;
            best_idx = i;
        }
    }
    return best_idx;
}

void SelectModel(int idx)
{
    if(idx == 0)
    {
        ArrayCopy(g_coeffs, g_coeffs_logreg);
        ArrayCopy(g_lot_coeffs, g_lot_coeffs_logreg);
        ArrayCopy(g_sl_coeffs, g_sl_coeffs_logreg);
        ArrayCopy(g_tp_coeffs, g_tp_coeffs_logreg);
        g_threshold = g_threshold_logreg;
        ArrayCopy(g_feature_mean, g_feature_mean_logreg);
        ArrayCopy(g_feature_std, g_feature_std_logreg);
        g_conformal_lower = g_conformal_lower_logreg;
        g_conformal_upper = g_conformal_upper_logreg;
    }
    else if(idx == 1)
    {
        ArrayCopy(g_coeffs, g_coeffs_xgboost);
        ArrayCopy(g_lot_coeffs, g_lot_coeffs_xgboost);
        ArrayCopy(g_sl_coeffs, g_sl_coeffs_xgboost);
        ArrayCopy(g_tp_coeffs, g_tp_coeffs_xgboost);
        g_threshold = g_threshold_xgboost;
        ArrayCopy(g_feature_mean, g_feature_mean_xgboost);
        ArrayCopy(g_feature_std, g_feature_std_xgboost);
        g_conformal_lower = g_conformal_lower_xgboost;
        g_conformal_upper = g_conformal_upper_xgboost;
    }
    else
    {
        ArrayCopy(g_coeffs, g_coeffs_lstm);
        ArrayCopy(g_lot_coeffs, g_lot_coeffs_lstm);
        ArrayCopy(g_sl_coeffs, g_sl_coeffs_lstm);
        ArrayCopy(g_tp_coeffs, g_tp_coeffs_lstm);
        g_threshold = g_threshold_lstm;
        ArrayCopy(g_feature_mean, g_feature_mean_lstm);
        ArrayCopy(g_feature_std, g_feature_std_lstm);
        g_conformal_lower = g_conformal_lower_lstm;
        g_conformal_upper = g_conformal_upper_lstm;
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
int    grpc_trade_queue_depth();
int    grpc_metric_queue_depth();
int    grpc_trade_retry_count();
int    grpc_metric_retry_count();
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

void LogUncertainDecision(int decision_id, double prob, double thr, string features, string action)
{
    int handle = FileOpen("uncertain_decisions.csv", FILE_READ|FILE_WRITE|FILE_CSV, ';');
    if(handle == INVALID_HANDLE)
        return;
    bool write_header = FileSize(handle) == 0;
    FileSeek(handle, 0, SEEK_END);
    if(write_header)
        FileWrite(handle, "decision_id", "action", "probability", "threshold", "features", "label");
    FileWrite(handle, decision_id, action, prob, thr, features, "");
    FileClose(handle);
}

void ReportWalSizes()
{
    QueueMetric("trade_wal_size=" + IntegerToString(GetWalSize(TRADE_WAL)));
    QueueMetric("metric_wal_size=" + IntegerToString(GetWalSize(METRIC_WAL)));
    QueueMetric("trade_queue_depth=" + IntegerToString(grpc_trade_queue_depth()));
    QueueMetric("metric_queue_depth=" + IntegerToString(grpc_metric_queue_depth()));
    QueueMetric("trade_retry_count=" + IntegerToString(grpc_trade_retry_count()));
    QueueMetric("metric_retry_count=" + IntegerToString(grpc_metric_retry_count()));
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
    int idx = RouteModel();
    SelectModel(idx);
    grpc_client_init(TRADE_HOST, TRADE_PORT, METRIC_HOST, METRIC_PORT);
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
}

void OnTimer()
{
    FlushTrades();
    FlushMetrics();
    grpc_flush();
    ReportWalSizes();
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

// __GET_FEATURE__


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

double TransformerScore()
{
    int W = g_transformer_window;
    int D = g_transformer_dim;
    int F = g_transformer_feat_dim;
    if(W <= 0 || D <= 0 || F <= 0)
        return 0.0;
    if(g_transformer_seq_len < W)
        return 0.0;
    double embed[]; ArrayResize(embed, W * D);
    for(int t = 0; t < W; t++)
    {
        for(int d = 0; d < D; d++)
        {
            double sum = g_embed_bias[d];
            for(int f = 0; f < F; f++)
            {
                int idx = d * F + f;
                sum += g_embed_weight[idx] * g_transformer_buffer[t * F + f];
            }
            embed[t * D + d] = sum;
        }
    }

    double Q[]; ArrayResize(Q, W * D);
    double K[]; ArrayResize(K, W * D);
    double V[]; ArrayResize(V, W * D);
    for(int t = 0; t < W; t++)
    {
        for(int d = 0; d < D; d++)
        {
            double sumq = 0, sumk = 0, sumv = 0;
            for(int j = 0; j < D; j++)
            {
                int idx = d * D + j;
                double val = embed[t * D + j];
                sumq += g_q_weight[idx] * val;
                sumk += g_k_weight[idx] * val;
                sumv += g_v_weight[idx] * val;
            }
            sumq += g_q_bias[d];
            sumk += g_k_bias[d];
            sumv += g_v_bias[d];
            Q[t * D + d] = sumq;
            K[t * D + d] = sumk;
            V[t * D + d] = sumv;
        }
    }

    double context[]; ArrayResize(context, W * D);
    for(int t = 0; t < W; t++)
    {
        double scores[]; ArrayResize(scores, W);
        double maxv = -1e10;
        for(int s = 0; s < W; s++)
        {
            double score = 0;
            for(int d = 0; d < D; d++)
                score += Q[t * D + d] * K[s * D + d];
            score /= MathSqrt(D);
            scores[s] = score;
            if(score > maxv) maxv = score;
        }
        double sumexp = 0;
        for(int s = 0; s < W; s++)
            sumexp += MathExp(scores[s] - maxv);
        for(int d = 0; d < D; d++)
            context[t * D + d] = 0.0;
        for(int s = 0; s < W; s++)
        {
            double w = MathExp(scores[s] - maxv) / sumexp;
            for(int d = 0; d < D; d++)
                context[t * D + d] += w * V[s * D + d];
        }
    }

    double pooled[]; ArrayResize(pooled, D);
    for(int d = 0; d < D; d++)
    {
        double sum = 0;
        for(int t = 0; t < W; t++)
            sum += context[t * D + d];
        pooled[d] = sum / W;
    }
    double z = g_out_bias;
    for(int d = 0; d < D; d++)
        z += g_out_weight[d] * pooled[d];
    return 1.0 / (1.0 + MathExp(-z));
}

double ScoreModel()
{
    if(g_use_transformer)
        return TransformerScore();
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
    int model_idx = RouteModel();
    SelectModel(model_idx);

    if(g_use_transformer)
    {
        int F = g_transformer_feat_dim;
        int W = g_transformer_window;
        if(ArraySize(g_transformer_buffer) != W * F)
            ArrayResize(g_transformer_buffer, W * F);
        for(int t = W - 1; t > 0; t--)
            for(int f = 0; f < F; f++)
                g_transformer_buffer[t * F + f] = g_transformer_buffer[(t - 1) * F + f];
        for(int f = 0; f < F; f++)
        {
            double val = GetFeature(f);
            double norm = (val - g_tfeature_mean[f]) / g_tfeature_std[f];
            g_transformer_buffer[f] = norm;
        }
        if(g_transformer_seq_len < W)
            g_transformer_seq_len++;
        Print("seq_len=", g_transformer_seq_len);
    }

    double prob = ScoreModel();
    double lot = PredictLot();
    double sl = PredictSLDistance();
    double tp = PredictTPDistance();
    g_decision_id++;
    int decision_id = g_decision_id;
    string decision = "hold";
    string reason = "";
    int order_type = GetOrderType(model_idx);
    double thr = OrderThreshold(order_type);
    bool uncertain = (prob >= g_conformal_lower && prob <= g_conformal_upper);
    if(uncertain)
    {
        decision = "skip";
        reason = "uncertain_prob";
    }
    else if(prob > thr && order_type == OP_BUY)
    {
        double sl_price = Ask - sl * Point;
        double tp_price = Ask + tp * Point;
        if(OrderSend(Symbol(), OP_BUY, lot, Ask, 3, sl_price, tp_price,
            "decision_id=" + IntegerToString(decision_id)) > 0)
        {
            decision = "buy";
        }
    }
    else if((1.0 - prob) > thr && order_type == OP_SELL)
    {
        double sl_price = Bid + sl * Point;
        double tp_price = Bid - tp * Point;
        if(OrderSend(Symbol(), OP_SELL, lot, Bid, 3, sl_price, tp_price,
            "decision_id=" + IntegerToString(decision_id)) > 0)
        {
            decision = "sell";
        }
    }
    string features = "";
    int feat_count = ArraySize(g_feature_mean);
    for(int i = 0; i < feat_count; i++)
    {
        if(i > 0)
            features += ":";
        features += DoubleToString(GetFeature(i), 8);
    }
    if(uncertain)
        LogUncertainDecision(decision_id, prob, thr, features, decision);
    QueueTrade(
        "decision_id=" + IntegerToString(decision_id) +
        ",decision=" + decision +
        ",prob=" + DoubleToString(prob, 8) +
        ",lot=" + DoubleToString(lot, 2) +
        ",sl=" + DoubleToString(sl, 2) +
        ",tp=" + DoubleToString(tp, 2) +
        ",reason=" + reason +
        ",features=" + features
    );
}
