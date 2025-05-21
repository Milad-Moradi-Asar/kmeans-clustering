# خوشه‌بندی با الگوریتم K-Means

در این پروژه دو مثال از الگوریتم خوشه‌بندی KMeans با استفاده از کتابخانه Scikit-Learn داریم:  
۱. خوشه‌بندی مشتریان فروشگاه  
۲. خوشه‌بندی مناطق مسکونی کالیفرنیا

---

## کتابخانه‌های استفاده‌شده

import pandas as pd  
import numpy as np  
from sklearn.cluster import KMeans  
import seaborn as sns  
import matplotlib.pyplot as plt  
from sklearn.datasets import fetch_california_housing  
from sklearn.preprocessing import StandardScaler  

---

## مثال اول: خوشه‌بندی مشتریان فروشگاه

df = pd.read_csv('Mall_Customers.csv')
بارگذاری دیتاست مشتریان از فایل CSV

km = KMeans(n_clusters=5)
km.fit(df[['Annual_Income_(k$)', 'Spending_Score_(1-100)']])
ساخت مدل خوشه‌بندی با ۵ دسته و آموزش آن بر اساس ستون‌های درآمد سالانه و امتیاز خرج‌کرد.

df['lable'] = km.labels_
افزودن برچسب خوشه‌ها به دیتافریم برای هر مشتری

sns.scatterplot(x=df['Annual_Income_(k$)'],
                y=df['Spending_Score_(1-100)'],
                hue=df['lable'],
                palette='summer')
رسم نمودار پراکندگی با رنگ‌های مختلف برای نمایش هر خوشه

---

## مثال دوم: خوشه‌بندی مناطق مسکونی کالیفرنیا

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
بارگیری دیتاست خانه‌های کالیفرنیا و تبدیل آن به جدول (DataFrame)

X_selected = X[['MedInc', 'AveRooms']]
انتخاب دو ویژگی اصلی برای خوشه‌بندی:  
درآمد متوسط (MedInc) و میانگین تعداد اتاق‌ها (AveRooms)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
استانداردسازی (نرمال‌سازی) ویژگی‌ها برای بهبود دقت خوشه‌بندی

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
ساخت مدل خوشه‌بندی با ۴ دسته و گرفتن برچسب هر نقطه

X['Cluster'] = clusters
افزودن برچسب خوشه به جدول داده‌ها

plt.figure(figsize=(8,6))
plt.scatter(X_selected['MedInc'], X_selected['AveRooms'], c=clusters, cmap='plasma')
plt.xlabel('درآمد متوسط')
plt.ylabel('میانگین تعداد اتاق‌ها')
plt.title('خوشه‌بندی مناطق مسکونی کالیفرنیا')
plt.show()
رسم نمودار نهایی برای نمایش خوشه‌ها با رنگ‌های متفاوت

---

## الگوریتم KMeans چیست؟

الگوریتم KMeans یکی از الگوریتم‌های معروف یادگیری ماشین برای خوشه‌بندی (Clustering) است.  
این الگوریتم داده‌ها را بر اساس شباهت به دسته‌هایی تقسیم می‌کند.  
هر خوشه دارای یک مرکز (center) است که نقاط به آن نزدیک‌تر هستند.

کاربردها:

- تحلیل رفتار مشتریان
- دسته‌بندی تصاویر یا داده‌ها
- خلاصه‌سازی داده‌ها
- پیشنهاددهی (Recommendation)
