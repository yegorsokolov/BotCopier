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
