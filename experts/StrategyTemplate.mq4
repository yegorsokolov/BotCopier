#property strict
#include "model_interface.mqh"

extern string SymbolToTrade = "EURUSD";
extern double Lots = 0.1;
extern int MagicNumber = 1234;

int OnInit()
{
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   // Placeholder for generated strategy logic
}

void OnDeinit(const int reason)
{
}
