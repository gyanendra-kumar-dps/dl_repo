import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
transactions = [['Burger', 'Ketchup', 'Fries'],['Burger', 'Ketchup'],['Burger', 'Fries'],['Pizza', 'Coke'],['Burger', 'Coke', 'Fries'],['Pizza', 'Burger', 'Ketchup'],['Burger', 'Ketchup', 'Coke'],['Fries', 'Ketchup'],['Pizza', 'Coke', 'Fries'],['Burger', 'Ketchup', 'Fries', 'Coke']]
print("Transactions:")
for t in transactions:
    print(t)
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)
print("\nEncoded Dataset:")
print(df)
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print("\nInterpretation Example:")
for index, row in rules.iterrows():
    antecedent = list(row['antecedents'])
    consequent = list(row['consequents'])
    confidence = round(row['confidence'] * 100, 2)
    print(f"If a customer buys {antecedent}, "
          f"they are {confidence}% likely to buy {consequent}")
plt.figure(figsize=(8, 5))
plt.scatter(
    rules['support'],
    rules['confidence'],
    s=rules['lift'] * 100,
    alpha=0.7,
    color='blue'
)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules: Support vs Confidence')
for i in range(len(rules)):
    plt.text(
        rules['support'].iloc[i],
        rules['confidence'].iloc[i],
        str(i)
    )
plt.grid(True)
plt.show()