#property strict

// Strategy template for generated expert advisor.

double GetFeature(int idx)
{
    switch(idx)
    {
    case 0: return MarketInfo(Symbol(), MODE_SPREAD); // spread
    case 1: return TimeHour(TimeCurrent()); // hour
    }
    return 0.0;
}

