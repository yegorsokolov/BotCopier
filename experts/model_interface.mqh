#ifndef __MODEL_INTERFACE_MQH__
#define __MODEL_INTERFACE_MQH__

struct ModelSignal
{
   datetime timestamp;
   string   symbol;
   int      direction; // 1 = buy, -1 = sell
   double   lots;
   double   price;
   double   sl;
   double   tp;
};

struct ModelMetrics
{
   string   model_id;
   int      predicted_events;
   int      matched_events;
   double   success_pct;
   double   coverage_pct;
};

// Description of a simple logistic regression model used by generated
// strategies.  ``coefficients`` is an array of weights that should be
// multiplied by the feature vector returned at runtime.  ``intercept``
// is added to the weighted sum before applying the logistic function.
struct LogisticModel
{
   double coefficients[];
   double intercept;
};

#endif
