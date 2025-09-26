from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from retrieve.nflfastr import get_sliding

sliding_df = get_sliding()

# Features and target
X = sliding_df[['week_t-3','week_t-2','week_t-1']]
y = sliding_df['y']

# Train/test split (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Train Ridge regression
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))