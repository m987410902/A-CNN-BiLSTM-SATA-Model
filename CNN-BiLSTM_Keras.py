# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply, Flatten, MaxPooling1D, Permute
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from bayes_opt import BayesianOptimization
from datetime import datetime
from tensorflow.keras.layers import Concatenate
from keras.layers import Lambda
from sklearn.metrics import mean_absolute_error
from keras.layers import Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Reshape
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed

# %%
# 设置全局参数
TIME_STEPS = 4
MODEL_PATH = './model.h5'  

# 函数：分离静态特征和动态特征，创造时间序列数据集（修改：移除季节编码相关代码）
def create_dataset_with_static_and_dynamic_features(dataset, static_feature_indices, look_back=TIME_STEPS):
    dynamic_X, static_X, dataY = [], [], []
    for i in range(0, len(dataset) - look_back + 1, look_back):
        dynamic_features = dataset[i:(i + look_back), 1:]  # 去掉目标列
        static_features = dataset[i, static_feature_indices]  # 多个静态特征
        target = dataset[i, 0]  # 目标值

        dynamic_X.append(dynamic_features)
        static_X.append(static_features)
        dataY.append(target)

    return np.array(dynamic_X), np.array(static_X), np.array(dataY)

# BiLSTM模型（修改：移除TPA相关代码，简化模型结构）
def build_model_with_cnn(lstm_units, dropout, dynamic_input_dims, static_input_dims):
    # 动态特征输入
    dynamic_inputs = Input(shape=(TIME_STEPS, dynamic_input_dims), name="dynamic_input")

    # CNN 层
    x = Conv1D(filters=64, kernel_size=4, strides=1, activation='relu', padding='same')(dynamic_inputs)
    x = Dropout(dropout)(x)

    # BiLSTM
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(dropout)(x)

    # 展平成一维向量
    x = Flatten()(x)

    # 静态特征输入
    static_inputs = Input(shape=(static_input_dims,), name="static_input")
    static_out = Dense(64, activation='relu')(static_inputs)
    static_out = Dropout(dropout)(static_out)

    # 动态和静态特征融合
    combined = Concatenate()([x, static_out])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(dropout)(combined)

    # 输出层
    output = Dense(1, activation='linear')(combined)

    # 构建模型
    model = Model(inputs=[dynamic_inputs, static_inputs], outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    return model

# 评估模型并计算 R² 和 RMSE
def evaluate_model(model, X, Y, dataset_type='Test'):
    results = model.predict(X).flatten()  # 确保输出为 1D
    r2 = r2_score(Y, results)
    rmse = np.sqrt(mean_squared_error(Y, results))
    print(f'{dataset_type} R^2: {r2:.4f}, RMSE: {rmse:.4f}')
    return results, r2, rmse

# 数据归一化和反归一化
def normalize_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def inverse_normalize(data, scaler):
    return scaler.inverse_transform(data)

# 绘制训练曲线
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if 'mae' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
    plt.tight_layout()
    plt.show()
    
# 反归一化预测值和实际值
def inverse_target_only(data, scaler):
    # 仅反归一化目标列，假设目标列是第一个
    dummy_data = np.zeros((len(data), scaler.scale_.shape[0]))
    dummy_data[:, 0] = data.flatten()
    return scaler.inverse_transform(dummy_data)[:, 0]

def inverse_and_export_results(predictions, actuals, scaler, output_path):
    # 反归一化
    predictions_inverse = inverse_target_only(predictions, scaler)
    actuals_inverse = inverse_target_only(actuals, scaler)
    
    # 创建 DataFrame 保存结果
    results_df = pd.DataFrame({
        'Predicted': predictions_inverse,
        'Actual': actuals_inverse
    })
    
    # 导出到 CSV 文件
    results_df.to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")

# %%
# 导入本地训练集和测试集数据
train_data = pd.read_excel(r"F:\Data\Study\DataSet\Project\1.Linan\M_Data\Prediction\La_2019_10m_Train.xlsx")
test_data = pd.read_excel(r"F:\Data\Study\DataSet\Project\1.Linan\M_Data\Prediction\La_2019_10m_Test.xlsx")
train_df = train_data.drop(['FID','YSSZ','TIME'], axis=1)
test_df = test_data.drop(['FID','YSSZ','TIME'], axis=1)

# 设置静态特征列表
static_features = ['Static_P_AGB']

# 从数据集中剔除静态特征用于随机森林
train_df_dynamic = train_df.drop(columns=static_features, errors='ignore')
test_df_dynamic = test_df.drop(columns=static_features, errors='ignore')

# 使用随机森林计算动态特征的重要性
target_column = 'AGB'  # 假设目标变量为 'AGB'
X_train_dynamic = train_df_dynamic.drop(columns=[target_column])
y_train = train_df_dynamic[target_column]

X_test_dynamic = test_df_dynamic.drop(columns=[target_column])
y_test = test_df_dynamic[target_column]

rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(X_train_dynamic, y_train)

# 提取特征重要性
feature_importances = pd.DataFrame({
    'Feature': X_train_dynamic.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 打印动态特征重要性
print("Feature Importance from Random Forest (Dynamic Features):")
print(feature_importances)

# 绘制特征重要性柱状图
plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances from Random Forest (Dynamic Features)')
plt.show()

# %%
# 保留完整数据
train_df_full = train_data.copy()
test_df_full = test_data.copy()

# 设置特征数量限制，选择重要性排序前 k 个的动态特征
top_k_features = 10  # 选择前 k 个特征
selected_dynamic_features = feature_importances['Feature'].iloc[:top_k_features].tolist()

# 将静态特征直接添加到筛选后的特征列表中
selected_features = selected_dynamic_features + static_features

# 打印最终筛选的特征
print("Final selected features:", selected_features)

# 构造训练和测试集
train_df = train_df[[target_column] + selected_features]
test_df = test_df[[target_column] + selected_features]

train_scaled, scaler = normalize_data(train_df.values)
test_scaled = scaler.transform(test_df.values)

# 获取静态特征索引（基于 selected_features 的索引）
static_feature_indices = [selected_features.index(f) + 1 for f in static_features]  # +1 是因为目标列在前

# 创建时间序列样本（修改：不再使用季节编码）
train_dynamic_X, train_static_X, train_Y = create_dataset_with_static_and_dynamic_features(
    train_scaled, static_feature_indices, TIME_STEPS
)
test_dynamic_X, test_static_X, test_Y = create_dataset_with_static_and_dynamic_features(
    test_scaled, static_feature_indices, TIME_STEPS
)

# 打印数据集形状
print(f'Train Dynamic X shape: {train_dynamic_X.shape}, Train Static X shape: {train_static_X.shape}, Train Y shape: {train_Y.shape}')
print(f'Test Dynamic X shape: {test_dynamic_X.shape}, Test Static X shape: {test_static_X.shape}, Test Y shape: {test_Y.shape}')

# 设置输入维度
INPUT_DIMS = len(selected_features)

# %%
# 定义贝叶斯优化目标函数的闭包（修改：移除季节编码相关参数）
def get_train_and_evaluate_with_data(train_dynamic_X, train_static_X, train_Y):
    def train_and_evaluate_model_with_data(lstm_units, dropout, batch_size, epochs):
        lstm_units = int(round(lstm_units))  # 确保 LSTM 单元数为整数
        batch_size = int(round(batch_size / 2) * 2)  # 批量大小调整为偶数
        epochs = int(round(epochs))  # 确保 epoch 数为整数
        dropout = round(dropout, 1)  # 保留一位小数
        K.clear_session()  # 清除当前会话，避免内存泄漏

        # 构建模型（修改：使用无TPA的模型）
        model = build_model_with_cnn(
            lstm_units=lstm_units,
            dropout=dropout,
            dynamic_input_dims=train_dynamic_X.shape[2],  # 动态特征维度
            static_input_dims=train_static_X.shape[1] if len(train_static_X.shape) > 1 else 1  # 静态特征维度
        )

        # 定义回调函数
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
        model_checkpoint = ModelCheckpoint(MODEL_PATH.format(datetime.now().strftime('%Y%m%d_%H%M%S')),
                                            save_best_only=True, monitor='val_loss', mode='min')

        # 模型训练（修改：不再使用季节编码）
        history = model.fit(
            [train_dynamic_X, train_static_X],  # 仅使用动态和静态特征
            train_Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping, model_checkpoint],
            verbose=0
        )
        val_loss = min(history.history['val_loss'])  # 获取验证集的最小损失值
        return -val_loss  # 贝叶斯优化需要最大化目标值，因此取负
    return train_and_evaluate_model_with_data

# 使用闭包获取目标函数（修改：不再传递季节编码）
train_and_evaluate_model_with_data = get_train_and_evaluate_with_data(train_dynamic_X, train_static_X, train_Y)

# 定义参数搜索空间，并为 batch_size 设置合理步长
pbounds = {
    'lstm_units': (32, 96),
    'dropout': (0.0, 0.2),  # 这里 dropout 固定为 0.0
    'batch_size': (32, 64),
    'epochs': (50, 100)
}

# 定义开关变量
use_bayesian_optimization = False  # 如果为 True，则使用贝叶斯优化；为 False 则使用自定义参数

if use_bayesian_optimization:
    # 执行贝叶斯优化
    optimizer = BayesianOptimization(
        f=train_and_evaluate_model_with_data,
        pbounds=pbounds,
        random_state=42
    )

    optimizer.maximize(init_points=3, n_iter=14)

    # 获取最佳参数
    best_params = optimizer.max['params']
    best_params['batch_size'] = int(round(best_params['batch_size'] / 2) * 2)
    best_params['epochs'] = int(round(best_params['epochs']))
    best_params['lstm_units'] = int(round(best_params['lstm_units']))
    best_params['dropout'] = round(best_params['dropout'], 1)

    print("Best Parameters from Bayesian Optimization:", best_params)
else:
    # 使用自定义参数
    best_params = {
        'lstm_units': 64,
        'dropout': 0.0,
        'batch_size': 64,
        'epochs': 100
    }
    print("Using custom parameters:", best_params)

# %%
# 设置动态和静态特征的输入维度
dynamic_input_dims = train_dynamic_X.shape[2]
static_input_dims = train_static_X.shape[1]  # 静态特征维度

# 构建模型（修改：使用无TPA的模型）
final_model = build_model_with_cnn(
    lstm_units=int(best_params['lstm_units']),
    dropout=round(best_params['dropout'], 1),
    dynamic_input_dims=dynamic_input_dims,
    static_input_dims=static_input_dims
)

# 模型训练（修改：不再使用季节编码）
final_history = final_model.fit(
    [train_dynamic_X, train_static_X],  # 仅使用动态和静态特征
    train_Y,
    epochs=int(best_params['epochs']),
    batch_size=int(best_params['batch_size']),
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=24, mode='min')],
    verbose=1
)

# 训练集和测试集评估（修改：不再使用季节编码）
train_results, train_r2, train_rmse = evaluate_model(
    final_model, [train_dynamic_X, train_static_X], train_Y, dataset_type='Train')

test_results, test_r2, test_rmse = evaluate_model(
    final_model, [test_dynamic_X, test_static_X], test_Y, dataset_type='Test')

# 调用 plot_history 显示训练过程中的损失曲线
plot_history(final_history)

# %%
# 模型预测（修改：不再使用季节编码）
train_predictions = final_model.predict([train_dynamic_X, train_static_X])
test_predictions = final_model.predict([test_dynamic_X, test_static_X])

# 反归一化训练集的预测结果和实际值
train_predictions_inverse = inverse_target_only(train_predictions, scaler)
train_actuals_inverse = inverse_target_only(train_Y, scaler)

# 反归一化测试集的预测结果和实际值
test_predictions_inverse = inverse_target_only(test_predictions, scaler)
test_actuals_inverse = inverse_target_only(test_Y, scaler)

# 计算反归一化后的 RMSE 和 MAE
train_rmse_inverse = np.sqrt(mean_squared_error(train_actuals_inverse, train_predictions_inverse))
train_mae_inverse = mean_absolute_error(train_actuals_inverse, train_predictions_inverse)
test_rmse_inverse = np.sqrt(mean_squared_error(test_actuals_inverse, test_predictions_inverse))
test_mae_inverse = mean_absolute_error(test_actuals_inverse, test_predictions_inverse)

# 打印反归一化后的性能指标
print(f'Train RMSE (Inverse Scaled): {train_rmse_inverse:.4f}')
print(f'Train MAE (Inverse Scaled): {train_mae_inverse:.4f}')
print(f'Test RMSE (Inverse Scaled): {test_rmse_inverse:.4f}')
print(f'Test MAE (Inverse Scaled): {test_mae_inverse:.4f}')

# 绘制测试集预测与真实值对比（反归一化后）
plt.figure(figsize=(7, 5))
plt.plot(test_predictions_inverse, label='Predicted - Test', color='blue')
plt.plot(test_actuals_inverse, label='Actual - Test', color='orange')
plt.title('Prediction vs Actual (Test Set) - Inverse Scaled')
plt.xlabel('Step')
plt.ylabel('Value')
plt.legend()
plt.show()

# 绘制测试集散点图（反归一化后）
plt.figure(figsize=(6, 5))
plt.scatter(test_predictions_inverse, test_actuals_inverse, label='Data Points', color='green', alpha=0.6)
sns.regplot(x=test_predictions_inverse, y=test_actuals_inverse, scatter=False, label='Fit Line', color='red')
plt.title('Fitted Curve: Predicted vs Actual (Test Set) - Inverse Scaled')
plt.xlabel('Predicted Value (Inverse Scaled)')
plt.ylabel('Actual Value (Inverse Scaled)')
plt.legend()
plt.show()




