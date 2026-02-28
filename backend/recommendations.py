import model as _model


def get_recommendations():
    _model.load_model()
    df = _model._data['df']

    target = 'Electricity Load'
    avg_load  = df[target].mean()
    peak_hour = df.groupby(df.index.hour)[target].mean().idxmax()
    off_hour  = df.groupby(df.index.hour)[target].mean().idxmin()
    peak_load = df.groupby(df.index.hour)[target].mean().max()
    off_load  = df.groupby(df.index.hour)[target].mean().min()

    # Seasonal analysis
    df_copy = df.copy()
    df_copy['month'] = df_copy.index.month
    summer_avg = df_copy[df_copy['month'].isin([6, 7, 8])][target].mean()
    winter_avg = df_copy[df_copy['month'].isin([12, 1, 2])][target].mean()

    # Weekend vs weekday
    df_copy['dow'] = df_copy.index.dayofweek
    weekday_avg = df_copy[df_copy['dow'] < 5][target].mean()
    weekend_avg = df_copy[df_copy['dow'] >= 5][target].mean()

    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2}
    recs = []

    # 1. Peak demand shifting
    peak_saving = (peak_load - avg_load) * 0.20
    recs.append({
        'priority':   'CRITICAL',
        'category':   '⚡ Peak Demand Shifting',
        'action':     f'Shift non-critical loads from {peak_hour}:00 (peak) to {off_hour}:00 (off-peak). '
                      f'Peak demand is {round(peak_load-avg_load, 1)} MW above average.',
        'saving_mw':  round(peak_saving, 1),
        'saving_pct': 20,
        'co2_kg_day': round(peak_saving * 0.386 * 24, 0),
    })

    # 2. Seasonal HVAC optimization
    if summer_avg > avg_load * 1.1:
        hvac_saving = (summer_avg - avg_load) * 0.15
        recs.append({
            'priority':   'HIGH',
            'category':   '🌡️ HVAC Pre-cooling Strategy',
            'action':     f'Summer load is {round((summer_avg/avg_load-1)*100,1)}% above annual avg. '
                          f'Pre-cool buildings 2h before peak to reduce reactive AC load.',
            'saving_mw':  round(hvac_saving, 1),
            'saving_pct': 15,
            'co2_kg_day': round(hvac_saving * 0.386 * 24, 0),
        })

    # 3. Weekend load optimization
    if weekday_avg > weekend_avg * 1.15:
        sched_saving = (weekday_avg - weekend_avg) * 0.10
        recs.append({
            'priority':   'HIGH',
            'category':   '📅 Smart Load Scheduling',
            'action':     f'Weekday load ({round(weekday_avg,1)} MW) is significantly higher than weekend '
                          f'({round(weekend_avg,1)} MW). Schedule industrial tasks to off-peak weekday hours.',
            'saving_mw':  round(sched_saving, 1),
            'saving_pct': 10,
            'co2_kg_day': round(sched_saving * 0.386 * 24, 0),
        })

    # 4. Demand response program
    dr_saving = avg_load * 0.08
    recs.append({
        'priority':   'HIGH',
        'category':   '🤝 Demand Response Program',
        'action':     f'Enroll large consumers in automated demand response. '
                      f'Target top 5% consumers ({round(df[target].quantile(0.95),1)} MW events) '
                      f'for curtailment during grid stress.',
        'saving_mw':  round(dr_saving, 1),
        'saving_pct': 8,
        'co2_kg_day': round(dr_saving * 0.386 * 24, 0),
    })

    # 5. Renewable integration
    renew_saving = avg_load * 0.12
    recs.append({
        'priority':   'MEDIUM',
        'category':   '🌱 Renewable Integration',
        'action':     f'AI forecast shows consistent {round(off_load,1)} MW baseline at {off_hour}:00. '
                      f'Optimal window for battery storage charging from renewable sources.',
        'saving_mw':  round(renew_saving, 1),
        'saving_pct': 12,
        'co2_kg_day': round(renew_saving * 0.386 * 24, 0),
    })

    # 6. Winter heating optimization
    if winter_avg > avg_load * 0.95:
        heat_saving = winter_avg * 0.07
        recs.append({
            'priority':   'MEDIUM',
            'category':   '❄️ Winter Heating Optimization',
            'action':     f'Winter avg load ({round(winter_avg,1)} MW). '
                          f'Deploy smart thermostats with AI scheduling — reduce heating load by 7% via setback strategies.',
            'saving_mw':  round(heat_saving, 1),
            'saving_pct': 7,
            'co2_kg_day': round(heat_saving * 0.386 * 24, 0),
        })

    # 7. Real-time anomaly response
    recs.append({
        'priority':   'MEDIUM',
        'category':   '🚨 Anomaly Auto-Response',
        'action':     f'GGCN model detects consumption anomalies in real-time. '
                      f'Automate isolation of fault zones when deviation exceeds 2σ from predicted load.',
        'saving_mw':  round(avg_load * 0.03, 1),
        'saving_pct': 3,
        'co2_kg_day': round(avg_load * 0.03 * 0.386 * 24, 0),
    })

    return sorted(recs, key=lambda x: priority_order.get(x['priority'], 3))