# Recalibration of conventional equations within the same participant-grouped CV
# New notebook cell: run after the cell that builds the main results (RES/OOF).
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
OOFc = OOF[FINAL]

hr0 = first.HR.loc[ix]; male = (1 - su.Sex)
D = pd.DataFrame({'Age': su.Age, 'BMI': bmi, 'Weight': su.Weight, 'Male': male, 'HR0': hr0, 'y': y}, index=ix)
D['abs'] = D.y * D.Weight / 1000  # absolute VO2max, L/min
D['ID'] = ID
D['minor'] = su.Age < 18

def metrics(name, pred, obs, kind):
    e = pred - obs
    return dict(Method=name, Type=kind, n=len(obs), R2=1 - (e**2).sum() / ((obs - obs.mean())**2).sum(),
                RMSE=np.sqrt((e**2).mean()), Bias=e.mean(), LoA_low=e.mean() - 1.96*e.std(ddof=1),
                LoA_high=e.mean() + 1.96*e.std(ddof=1), Within_1MET=100*(np.abs(e) <= MET).mean())

def recal_oof(d, feats, target='y', to_rel=None):
    """Refit the equation's predictor set (OLS) in training folds; predict held-out participants."""
    p = np.full(len(d), np.nan)
    for tr, te in GroupKFold(5, shuffle=True, random_state=SEED).split(d, groups=d.ID):
        m = LinearRegression().fit(d.iloc[tr][feats], d.iloc[tr][target])
        p[te] = m.predict(d.iloc[te][feats])
    return p if to_rel is None else p * 1000 / d.Weight.values

# Original (published coefficients) predictions, as in the notebook
orig = {'Jackson et al. (1990)': 56.363 + 1.921*7 - .381*D.Age - .754*D.BMI + 10.987*D.Male,
        'Jurca et al. (2005)': 3.5*(18.07 + 2.77*D.Male - .10*D.Age - .17*D.BMI - .03*D.HR0 + 3.03),
        'Rexhepi & Brestovci (2014)': (3.542 - .014*D.Age + .015*D.Weight - .011*D.HR0) * 1000 / D.Weight,
        'Jalili et al. (2023)': 62.914 + .875*D.Age - 1.156*D.BMI - .12*D.HR0}

rows = []
# --- Full cohort (981): Jackson original + recalibrated
rows.append(metrics('Jackson et al. (1990)', orig['Jackson et al. (1990)'].values, D.y.values, 'Original'))
rows.append(metrics('Jackson et al. (1990)', recal_oof(D, ['Age', 'BMI', 'Male']), D.y.values, 'Recalibrated'))
# --- Tests with initial HR (738): Jurca, Rexhepi original + recalibrated
H = D[D.HR0.notna()]
for n, f, kw in [('Jurca et al. (2005)', ['Age', 'BMI', 'Male', 'HR0'], {}),
                 ('Rexhepi & Brestovci (2014)', ['Age', 'Weight', 'HR0'], dict(target='abs', to_rel=True))]:
    rows.append(metrics(n, orig[n].loc[H.index].values, H.y.values, 'Original'))
    rows.append(metrics(n, recal_oof(H, f, **kw), H.y.values, 'Recalibrated'))
# --- Jalili: only tests from participants <18 y with initial HR (its source population)
J = D[D.minor & D.HR0.notna() & (D.Male == 1)]
rows.append(metrics('Jalili et al. (2023)', orig['Jalili et al. (2023)'].loc[J.index].values, J.y.values, 'Original, boys <18 y'))
rows.append(metrics('Jalili et al. (2023)', recal_oof(J, ['Age', 'BMI', 'HR0']), J.y.values, 'Recalibrated, boys <18 y'))

# --- Study models on the same test subsets (out-of-fold predictions from the main analysis)
pos = {t: i for i, t in enumerate(ix)}
for subset_name, S in [('all', D), ('HR0', H), ('minor_HR0', J)]:
    k = np.array([pos[t] for t in S.index])
    for r, m in [('Instant_full', 'MLR'), ('Simple', 'XGBoost'), ('Instant0', 'MLR')]:
        rows.append(metrics(f'{r} - {m}', OOFc[r, m][k], yt[k], f'This study [{subset_name}]'))

T = pd.DataFrame(rows)
print('n minors with HR0:', len(J), 'participants:', J.ID.nunique(), '| minors total tests', D.minor.sum())
T.round(3).to_csv('tables/Table7_recalibrated.csv', index=False)
T.round(3)
