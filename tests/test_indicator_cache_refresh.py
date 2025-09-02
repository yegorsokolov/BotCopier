from pathlib import Path


def test_strategy_template_has_refresh_flag():
    content = Path("experts/StrategyTemplate.mq4").read_text().splitlines()
    assert any("CachedFeatures" in line for line in content)
    assert any("NeedsFeatureRefresh" in line for line in content)
    on_tick_idx = next(i for i, line in enumerate(content) if "void OnTick()" in line)
    assert any("if(Time[0] != LastFeatureTime)" in line for line in content[on_tick_idx:on_tick_idx + 5])
    assert any("NeedsFeatureRefresh = true;" in line for line in content[on_tick_idx:on_tick_idx + 10])
    refresh_idx = next(i for i, line in enumerate(content) if "void RefreshIndicatorCache()" in line)
    assert any("if(!NeedsFeatureRefresh)" in line for line in content[refresh_idx:refresh_idx + 5])


def test_refresh_indicator_cache_once_per_bar():
    class Dummy:
        def __init__(self):
            self.needs_refresh = True
            self.last_time = 0
            self.current_time = 0
            self.refresh_count = 0

        def RefreshIndicatorCache(self):
            if not self.needs_refresh:
                return
            self.needs_refresh = False
            self.last_time = self.current_time
            self.refresh_count += 1

        def GetFeature(self):
            self.RefreshIndicatorCache()
            return self.refresh_count

        def OnTick(self, t):
            self.current_time = t
            if t != self.last_time:
                self.needs_refresh = True

    s = Dummy()
    s.OnTick(1)
    s.GetFeature()
    s.GetFeature()
    assert s.refresh_count == 1
    s.OnTick(1)
    s.GetFeature()
    assert s.refresh_count == 1
    s.OnTick(2)
    s.GetFeature()
    assert s.refresh_count == 2
