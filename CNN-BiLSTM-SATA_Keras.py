# %%
# AGB预测模型：结合CNN、BiLSTM和注意力机制
# 功能：使用时间序列数据预测生物量(AGB)，包含特征重要性分析和注意力权重可视化

# ----------------------------
# 1. 导入必要的库
# ----------------------------
# 基础数据处理库
import numpy as np
import pandas as pd
import random
from datetime import datetime

# 可视化库
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.patches import Rectangle, FancyArrow

# 机器学习库               
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# 深度学习库
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, 
                                     Multiply, Flatten, MaxPooling1D, Permute, Concatenate,
                                     Lambda, Activation, Embedding, Reshape, TimeDistributed)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

# 其他工具
from bayes_opt import BayesianOptimization
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
# ----------------------------
# 2. 全局设置与参数配置
# ----------------------------
# 可视化参数
plt.rcParams['figure.dpi'] = 150
plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 11

# 模型参数
TIME_STEPS = 4
MODEL_PATH = './model_{}.h5'  # 添加时间戳占位符
TARGET_COLUMN = 'AGB'         # 目标变量列名
STATIC_FEATURES = ['Static_P_AGB']  # 静态特征列表
TOP_K_FEATURES = 10           # 选择的动态特征数量

# %%
# ----------------------------
# 3. 数据处理函数
# ----------------------------
def create_dataset_with_static_and_dynamic_features(dataset, static_feature_indices, look_back=TIME_STEPS):
    """
    分离静态特征和动态特征，创造时间序列数据集
    
    参数:
        dataset: 输入数据集
        static_feature_indices: 静态特征的索引列表
        look_back: 时间步长，默认为全局参数TIME_STEPS
        
    返回:
        动态特征、静态特征、季节特征和目标值的数组
    """
    dynamic_X, static_X, season_X, dataY = [], [], [], []
    for i in range(0, len(dataset) - look_back + 1, look_back):
        season_index = -1  # 最后一列是季节编码
        # 提取动态特征（去掉目标列和季节列）
        dynamic_features = dataset[i:(i + look_back), 1:season_index]
        # 提取静态特征
        static_features = dataset[i, static_feature_indices]
        # 提取目标值
        target = dataset[i, 0]
        # 提取季节编码（整数类型）
        season_codes = dataset[i:(i + look_back), season_index].astype(int)
        
        dynamic_X.append(dynamic_features)
        static_X.append(static_features)
        season_X.append(season_codes)
        dataY.append(target)
    
    return np.array(dynamic_X), np.array(static_X), np.array(season_X), np.array(dataY)

def normalize_data(data):
    """数据归一化处理"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def inverse_normalize(data, scaler):
    """数据反归一化处理"""
    return scaler.inverse_transform(data)

def inverse_target_only(data, scaler):
    """仅对目标列进行反归一化（假设目标列是第一列）"""
    dummy_data = np.zeros((len(data), scaler.scale_.shape[0]))
    dummy_data[:, 0] = data.flatten()
    return scaler.inverse_transform(dummy_data)[:, 0]

def inverse_and_export_results(predictions, actuals, scaler, output_path):
    """反归一化并导出预测结果和实际值到CSV文件"""
    # 反归一化
    predictions_inverse = inverse_target_only(predictions, scaler)
    actuals_inverse = inverse_target_only(actuals, scaler)
    
    # 创建DataFrame保存结果
    results_df = pd.DataFrame({
        'Predicted': predictions_inverse,
        'Actual': actuals_inverse
    })
    
    # 导出到CSV文件
    results_df.to_csv(output_path, index=False)
    print(f"结果已导出至: {output_path}")

# ----------------------------
# 4. 模型构建函数
# ----------------------------
def temporal_attention(inputs, season_embedding):
    """
    时间注意力机制：确保每个样地的四个季节共享相同的权重
    
    参数:
        inputs: 输入特征，形状为(batch_size, time_steps=4, features)
        season_embedding: 季节嵌入向量，形状为(batch_size, time_steps=8)
        
    返回:
        weighted_output: 加权后的输出（与inputs形状相同）
        attention_weights: 注意力权重（batch_size, time_steps）
    """
    # 拼接BiLSTM输出与季节嵌入，作为注意力得分的输入
    combined = Concatenate(axis=-1)([inputs, season_embedding])  # (batch_size, 4, features + 8)
    
    # 注意力得分计算
    scores = Dense(1, activation='tanh')(combined)  # (batch_size, 4, 1)
    scores = Lambda(lambda x: K.squeeze(x, axis=-1))(scores)  # (batch_size, 4)
    
    # Softmax归一化得到注意力权重
    attention_weights = Activation('softmax', name='time_attention_weights')(scores)  # (batch_size, 4)
    
    # 加权动态特征
    attention_weights_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(attention_weights)  # (batch_size, 4, 1)
    weighted_output = Multiply()([inputs, attention_weights_expanded])  # (batch_size, 4, features)
    
    return weighted_output, attention_weights

def build_model_with_cnn_and_temporal_attention(lstm_units, dropout, dynamic_input_dims, static_input_dims):
    """
    构建包含CNN、BiLSTM和时间注意力机制的模型
    
    参数:
        lstm_units: LSTM单元数量
        dropout: dropout比率
        dynamic_input_dims: 动态特征维度
        static_input_dims: 静态特征维度
        
    返回:
        构建好的Keras模型
    """
    # 动态特征输入
    dynamic_inputs = Input(shape=(TIME_STEPS, dynamic_input_dims), name="dynamic_input")
    
    # 季节编码输入（整型，范围0-3）
    season_inputs = Input(shape=(TIME_STEPS,), dtype='int32', name="season_input")  # (batch, 4)
    
    # 季节嵌入层（将0~3编码为8维向量）
    season_embedding = Embedding(input_dim=4, output_dim=8, input_length=TIME_STEPS)(season_inputs)  # (batch, 4, 8)
    
    # 合并动态特征和季节嵌入
    x = Concatenate(axis=-1)([dynamic_inputs, season_embedding])  # (batch, 4, dynamic_dims + 8)
    
    # CNN层
    x = Conv1D(filters=64, kernel_size=4, strides=1, activation='relu', padding='same')(x)
    x = Dropout(dropout)(x)
    
    # BiLSTM
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(dropout)(x)
    
    # 应用TPA注意力机制（加入季节嵌入）
    attention_output, time_attention_weights = temporal_attention(x, season_embedding)
    
    # 展平成一维向量
    x = Flatten()(attention_output)
    
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
    model = Model(inputs=[dynamic_inputs, static_inputs, season_inputs], outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    
    return model

# ----------------------------
# 5. 模型训练与评估函数
# ----------------------------
def evaluate_model(model, X, Y, dataset_type='Test'):
    """
    评估模型性能并计算R²和RMSE
    
    参数:
        model: 训练好的模型
        X: 输入特征
        Y: 目标值
        dataset_type: 数据集类型（如'Train'或'Test'）
        
    返回:
        预测结果、R²值和RMSE值
    """
    results = model.predict(X).flatten()  # 确保输出为1D
    r2 = r2_score(Y, results)
    rmse = np.sqrt(mean_squared_error(Y, results))
    print(f'{dataset_type} R^2: {r2:.4f}, RMSE: {rmse:.4f}')
    return results, r2, rmse

def get_train_and_evaluate_with_data(train_dynamic_X, train_static_X, train_season_codes, train_Y):
    """
    定义贝叶斯优化目标函数的闭包
    
    参数:
        train_dynamic_X: 训练集动态特征
        train_static_X: 训练集静态特征
        train_season_codes: 训练集季节编码
        train_Y: 训练集目标值
        
    返回:
        用于贝叶斯优化的目标函数
    """
    def train_and_evaluate_model_with_data(lstm_units, dropout, batch_size, epochs):
        # 确保参数为合适的类型
        lstm_units = int(round(lstm_units))
        batch_size = int(round(batch_size / 2) * 2)  # 批量大小调整为偶数
        epochs = int(round(epochs))
        dropout = round(dropout, 1)
        
        K.clear_session()  # 清除当前会话，避免内存泄漏
        
        # 构建模型
        model = build_model_with_cnn_and_temporal_attention(
            lstm_units=lstm_units,
            dropout=dropout,
            dynamic_input_dims=train_dynamic_X.shape[2],
            static_input_dims=train_static_X.shape[1] if len(train_static_X.shape) > 1 else 1
        )
        
        # 定义回调函数
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_checkpoint = ModelCheckpoint(MODEL_PATH.format(timestamp),
                                           save_best_only=True, monitor='val_loss', mode='min')
        
        # 模型训练
        history = model.fit(
            [train_dynamic_X, train_static_X, train_season_codes],
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

# ----------------------------
# 6. 可视化函数
# ----------------------------
def plot_history(history):
    """绘制训练过程中的损失和MAE曲线"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制MAE曲线（如果存在）
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

def plot_attention_mean_std(model, dynamic_X, static_X, season_X,
                           selected_indices=None, dataset_name="Test",
                           font_size=10, title_font_size=12, label_font_size=10):
    """
    对选中样地的时间注意力权重进行均值±标准差可视化
    """
    print(f"最终选取的样地数量: {len(selected_indices)}")
    
    # 获取时间注意力层输出
    attention_layer = model.get_layer("time_attention_weights")
    attention_model = Model(inputs=model.input[0:3], outputs=attention_layer.output)
    attention_weights = attention_model.predict([dynamic_X, static_X, season_X])
    attention_weights = attention_weights.reshape(attention_weights.shape[0], -1)  # (样地数, 4)
    
    # 默认取前30个样地
    if selected_indices is None:
        selected_indices = list(range(min(30, len(attention_weights))))
    
    selected_weights = attention_weights[selected_indices]
    mean_weights = selected_weights.mean(axis=0)
    std_weights = selected_weights.std(axis=0)
    
    # 可视化
    x = np.arange(4)
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    plt.figure(figsize=(6, 4))
    plt.errorbar(x, mean_weights, yerr=std_weights, fmt='-o',
                capsize=5, color='teal', ecolor='gray')
    plt.ylim(0.1, 0.4)
    
    # 设置坐标轴
    plt.xticks(x, seasons, fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title(f'Temporal Attention Weights (Lin’an)', 
              fontsize=title_font_size, fontweight='normal')
    plt.xlabel('Season', fontsize=label_font_size)
    plt.ylabel('Attention Weight', fontsize=label_font_size)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    # 输出注意力权重统计
    for i, season in enumerate(seasons):
        print(f"{season}: Mean = {mean_weights[i]:.4f}, Std = {std_weights[i]:.4f}, "
              f"Range ≈ [{mean_weights[i]-std_weights[i]:.4f}, {mean_weights[i]+std_weights[i]:.4f}]")

def visualize_attention_weights(model, dynamic_X, static_X, season_codes, dataset_name='Test'):
    """可视化时间注意力权重，包括折线图和热力图"""
    # 获取注意力权重层
    time_attention_layer = model.get_layer('time_attention_weights')
    
    # 构建模型用于提取时间注意力权重
    time_attention_model = Model(inputs=model.input[0:3], outputs=time_attention_layer.output)
    
    # 获取注意力权重 (batch, time_steps)
    time_attention_weights = time_attention_model.predict([dynamic_X, static_X, season_codes])
    time_attention_weights = time_attention_weights.reshape(time_attention_weights.shape[0], -1)
    
    # 折线图可视化前10个样本
    plt.figure(figsize=(9, 6))
    for i in range(min(10, len(time_attention_weights))):
        plt.plot(range(1, time_attention_weights.shape[1] + 1),
                 time_attention_weights[i], marker='o', label=f'Sample {i+1}')
    plt.title(f'Time Attention Weights Visualization ({dataset_name})')
    plt.xlabel('Time Steps')
    plt.ylabel('Attention Weight')
    
    # 保留4位小数的y轴格式
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.4f}'))
    
    plt.xticks(range(1, time_attention_weights.shape[1] + 1),
               ['Spring', 'Summer', 'Autumn', 'Winter'])  # 替换x轴标签为季节
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 热力图可视化（前10个样本），保留4位小数
    plt.figure(figsize=(8, 6))
    sns.heatmap(time_attention_weights[:10],
                annot=True,
                fmt=".4f",  # 设置格式为4位小数
                cmap='YlGnBu',
                cbar=True)
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=['Spring', 'Summer', 'Autumn', 'Winter'], rotation=0)
    plt.yticks(ticks=np.arange(0.5, 10.5), labels=[f'Sample {i+1}' for i in range(10)], rotation=0)
    plt.title(f'Temporal Attention Weights Heatmap ({dataset_name})')
    plt.xlabel('Season')
    plt.ylabel('Sample')
    plt.tight_layout()
    plt.show()
    
    # 打印注意力权重的具体值（保留4位小数）
    print(f"Time Attention Weights ({dataset_name}):")
    for i in range(min(10, len(time_attention_weights))):
        formatted_weights = [f'{w:.4f}' for w in time_attention_weights[i]]
        print(f"Sample {i+1}: {formatted_weights}")

# %%
# ----------------------------
# 7. 主程序
# ----------------------------
if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)
    
    # ----------------------------
    # 7.1 数据加载与预处理
    # ----------------------------
    # 导入本地训练集和测试集数据
    train_data = pd.read_excel(r"F:\Data\Study\DataSet\Project\1.Linan\M_Data\Prediction\La_2019_10m_Train.xlsx")
    test_data = pd.read_excel(r"F:\Data\Study\DataSet\Project\1.Linan\M_Data\Prediction\La_2019_10m_Test.xlsx")
    
    # 数据预处理：删除不需要的列
    train_df = train_data.drop(['FID','YSSZ','TIME'], axis=1)
    test_df = test_data.drop(['FID','YSSZ','TIME'], axis=1)
        
    # 保留完整数据（包含所有列，包括Season_Index）
    train_df_full = train_data.copy()
    test_df_full = test_data.copy()
        
    # ----------------------------
    # 7.2 特征重要性分析（使用随机森林）
    # ----------------------------
    # 从数据集中剔除静态特征用于随机森林
    train_df_dynamic = train_df.drop(columns=STATIC_FEATURES, errors='ignore')
    test_df_dynamic = test_df.drop(columns=STATIC_FEATURES, errors='ignore')
    
    # 准备随机森林的输入
    X_train_dynamic = train_df_dynamic.drop(columns=[TARGET_COLUMN])
    y_train = train_df_dynamic[TARGET_COLUMN]
    X_test_dynamic = test_df_dynamic.drop(columns=[TARGET_COLUMN])
    y_test = test_df_dynamic[TARGET_COLUMN]
    
    # 训练随机森林模型
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X_train_dynamic, y_train)
    
    # 提取并可视化特征重要性
    feature_importances = pd.DataFrame({
        'Feature': X_train_dynamic.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("特征重要性（随机森林）:")
    print(feature_importances)
    
    # 绘制特征重要性柱状图
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('特征重要性（随机森林）')
    plt.tight_layout()
    plt.show()

# %%
# 构建前10个重要特征的图表
top_features = feature_importances.head(TOP_K_FEATURES)
print(f"\n前{TOP_K_FEATURES}个重要特征:")
print(top_features)

plt.figure(figsize=(3, 2.5), dpi=200)
ax = sns.barplot(x='Feature', y='Importance', data=top_features, color='teal')

# 设置标题与坐标轴
plt.title('Seasonal Module Feature Importance (Lin’an)', fontsize=7.5, fontweight='light')
plt.xlabel('', fontsize=5.5)
plt.ylabel('', fontsize=5.5)
ax.tick_params(axis='x', labelsize=5.5, rotation=45)  # 旋转特征名，防止重叠
ax.tick_params(axis='y', labelsize=5.5)

# 网格线 + 布局
ax.grid(True, linestyle='', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
# ----------------------------
# 7.3 数据准备与划分
# ----------------------------
# 选择特征：前K个动态特征 + 静态特征
selected_dynamic_features = feature_importances['Feature'].iloc[:TOP_K_FEATURES].tolist()
selected_features = selected_dynamic_features + STATIC_FEATURES
print("\n最终选择的特征:", selected_features)

# 提取季节编码列（未归一化）
train_season_series = train_df_full['Season_Index'].values
test_season_series = test_df_full['Season_Index'].values

# 构造训练和测试集（删去Season_Index后归一化）
train_df_selected = train_df[[TARGET_COLUMN] + selected_features]
test_df_selected = test_df[[TARGET_COLUMN] + selected_features]

# 数据归一化
train_scaled, scaler = normalize_data(train_df_selected.values)
test_scaled = scaler.transform(test_df_selected.values)

# 获取静态特征索引（基于selected_features的索引）
static_feature_indices = [selected_features.index(f) + 1 for f in STATIC_FEATURES]  # +1是因为目标列在前

# 创建时间序列样本
train_dynamic_X, train_static_X, train_season_X, train_Y = create_dataset_with_static_and_dynamic_features(
    train_scaled, static_feature_indices, TIME_STEPS
)
test_dynamic_X, test_static_X, test_season_X, test_Y = create_dataset_with_static_and_dynamic_features(
    test_scaled, static_feature_indices, TIME_STEPS
)
    
# 将season_index reshape成(样本数, TIME_STEPS)
train_season_X = train_season_series.reshape(-1, TIME_STEPS)
test_season_X = test_season_series.reshape(-1, TIME_STEPS)

# 打印数据集形状
print(f'\n训练集动态特征形状: {train_dynamic_X.shape}, 训练集静态特征形状: {train_static_X.shape}, 训练集目标值形状: {train_Y.shape}')
print(f'测试集动态特征形状: {test_dynamic_X.shape}, 测试集静态特征形状: {test_static_X.shape}, 测试集目标值形状: {test_Y.shape}')

# %%
# ----------------------------
# 7.4 模型训练（贝叶斯优化或固定参数）
# ----------------------------
# 定义参数搜索空间
pbounds = {
    'lstm_units': (32, 96),
    'dropout': (0.0, 0.2),
    'batch_size': (32, 64),
    'epochs': (50, 100),
    'learning_rate': (-4, -2)
}
    
# 定义开关变量：是否使用贝叶斯优化
use_bayesian_optimization = False  # True: 使用贝叶斯优化；False: 使用自定义参数

if use_bayesian_optimization:
    # 执行贝叶斯优化
    train_and_evaluate_model_with_data = get_train_and_evaluate_with_data(
        train_dynamic_X, train_static_X, train_season_X, train_Y
    )
        
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
    
    print("\n贝叶斯优化得到的最佳参数:", best_params)
else:
    # 使用自定义参数
    best_params = {
        'lstm_units': 64,
        'dropout': 0.0,
        'batch_size': 64,
        'epochs': 100
        'learning_rate': -3
    }
    print("\n使用自定义参数:", best_params)

# %%
# ----------------------------
# 7.5 最终模型训练与评估
# ----------------------------
# 设置输入维度
dynamic_input_dims = train_dynamic_X.shape[2]
static_input_dims = train_static_X.shape[1]  # 静态特征维度

# 构建最终模型
final_model = build_model_with_cnn_and_temporal_attention(
    lstm_units=int(best_params['lstm_units']),
    dropout=round(best_params['dropout'], 1),
    dynamic_input_dims=dynamic_input_dims,
    static_input_dims=static_input_dims
)

# 模型训练
final_history = final_model.fit(
    [train_dynamic_X, train_static_X, train_season_X],
    train_Y,
    epochs=int(best_params['epochs']),
    batch_size=int(best_params['batch_size']),
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=24, mode='min')],
    verbose=1
)

# 训练集和测试集评估
print("\n模型评估结果:")
train_results, train_r2, train_rmse = evaluate_model(
    final_model, [train_dynamic_X, train_static_X, train_season_X], train_Y, dataset_type='Train')

test_results, test_r2, test_rmse = evaluate_model(
    final_model, [test_dynamic_X, test_static_X, test_season_X], test_Y, dataset_type='Test')

# 绘制训练过程曲线
plot_history(final_history)

# %%
# ----------------------------
# 7.6 结果分析与可视化
# ----------------------------
# 模型预测
train_predictions = final_model.predict([train_dynamic_X, train_static_X, train_season_X])
test_predictions = final_model.predict([test_dynamic_X, test_static_X, test_season_X])

# 反归一化
train_predictions_inverse = inverse_target_only(train_predictions, scaler)
train_actuals_inverse = inverse_target_only(train_Y, scaler)
test_predictions_inverse = inverse_target_only(test_predictions, scaler)
test_actuals_inverse = inverse_target_only(test_Y, scaler)

# 计算反归一化后的性能指标
train_rmse_inverse = np.sqrt(mean_squared_error(train_actuals_inverse, train_predictions_inverse))
train_mae_inverse = mean_absolute_error(train_actuals_inverse, train_predictions_inverse)
test_rmse_inverse = np.sqrt(mean_squared_error(test_actuals_inverse, test_predictions_inverse))
test_mae_inverse = mean_absolute_error(test_actuals_inverse, test_predictions_inverse)

# 打印反归一化后的性能指标
print(f'\n反归一化后的性能指标:')
print(f'Train RMSE: {train_rmse_inverse:.4f}')
print(f'Train MAE: {train_mae_inverse:.4f}')
print(f'Test RMSE: {test_rmse_inverse:.4f}')
print(f'Test MAE: {test_mae_inverse:.4f}')

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
plt.title('Fitted Curve: Predicted vs Actual (Test Set)')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.legend()
plt.show()

# %%
# ----------------------------
# 7.7 注意力权重分析
# ----------------------------
# 分层抽样样地（低中高各100个）
test_df_sample = pd.read_excel(r'F:\Data\Study\DataSet\Project\1.Linan\M_Data\Prediction\La_2019_10m_Test.xlsx')
agb_values = test_df_sample['AGB'].values[::4]  # 每4行为一个样地
sample_count = test_dynamic_X.shape[0]
agb_values = agb_values[:sample_count]  # 对齐样地数量

# 分层阈值
low_thres = np.quantile(agb_values, 0.33)
high_thres = np.quantile(agb_values, 0.66)

low_indices = np.where(agb_values <= low_thres)[0]
mid_indices = np.where((agb_values > low_thres) & (agb_values <= high_thres))[0]
high_indices = np.where(agb_values > high_thres)[0]

# 随机选择样本
selected_sample_indices = np.concatenate([
    np.random.choice(low_indices, size=min(100, len(low_indices)), replace=False),
    np.random.choice(mid_indices, size=min(100, len(mid_indices)), replace=False),
    np.random.choice(high_indices, size=min(100, len(high_indices)), replace=False)
])

# 可视化注意力权重均值和标准差
plot_attention_mean_std(
    final_model,
    test_dynamic_X,
    test_static_X,
    test_season_X,
    selected_indices=selected_sample_indices,
    dataset_name="Test",
    font_size=11,
    title_font_size=13,
    label_font_size=11,
)

# 可视化注意力权重
#visualize_attention_weights(
    #final_model,
    #test_dynamic_X,
    #test_static_X,
    #test_season_X,
    #dataset_name='Test'
#)

# %%



