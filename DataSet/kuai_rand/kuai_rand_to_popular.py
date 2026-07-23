import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# item = pd.read_csv(r"C:\Code\Frontdoor\SDR\DataSet\kuai_rand\train.csv").to_numpy().astype(float)

train_df = pd.read_csv(r"C:\Code\Frontdoor\SDR\DataSet\kuai_rand\train.csv")

print(len(train_df))

item_counts = train_df['item_id'].value_counts().reset_index()
item_counts.columns = ['item_id', 'count']

print(item_counts, item_counts.count())

save_sorted_counts = item_counts.sort_values('item_id', ascending=True).reset_index(drop=True)

# save_sorted_counts = save_sorted_counts.drop('item_id',axis=1)
print(save_sorted_counts)
# save_sorted_counts.to_csv('item_counts.csv', index=False)

# 对计数进行排序（从高到低）
sorted_counts = item_counts.sort_values('count', ascending=False).reset_index(drop=True)

pop= sorted_counts[sorted_counts['count'] < 210]['count'].sum()
unpop =sorted_counts[sorted_counts['count'] >= 210]['count'].sum()
print(pop,unpop,pop / (pop + unpop), sorted_counts['count'].mean())


plt.figure(figsize=(15, 10))

# 1. 热门商品分布图 (前20名)
plt.subplot(2, 2, 1)
plt.plot(sorted_counts['count'])
# sns.barplot(x='count', y='item_id', data=top_items, palette='viridis')
plt.show()

