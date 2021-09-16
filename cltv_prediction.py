import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
# Gerekli kütüphaneleri import ediyoruz

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
#Görünüm olarak daha iyi durması için ayarlamalar yapıyoruz.

#######################
# Verinin excel'den okunması
#######################

df_ = pd.read_excel(r"C:\Users\ALPEREN MİNNET\Desktop\DSMLBC\datasets\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df = df_.copy() # Verinin kopyasını oluşturalım.

#######################
# Veri Ön İşleme
#######################

df = df[df["Country"] == "United Kingdom"] #UK olanları aldık.
df.isnull().sum()
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
# Gerekli ön işlemeleri yaptık.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

#Veride aykırı değerler olduğu için aykırı değerleri törpüledik.

df["TotalPrice"] = df["Quantity"] * df["Price"] #Müşterinin faturasında toplam ne kadar ücret ödediğini bulduk.

today_date = dt.datetime(2011, 12, 11)

#######################
# Lifetime veri yapısının hazırlanması
#######################

# recency: Son satın alma üzerinden geçen zaman. Haftalık.
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(1)
# değişkenlerin isimlendirilmesi
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# monetary değerinin satın alma başına ortalama kazanç olarak ifade edilmesi
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# monetary sıfırdan büyük olanların seçilmesi
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df.head()

# BGNBD için recency ve T'nin haftalık cinsten ifade edilmesi
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# frequency'nin 1'den büyük olması gerekmektedir.
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

#######################
# BG/NBD Modelinin Kurulması
#######################

bgf = BetaGeoFitter(penalizer_coef=0.001) #Modeli kurduk
#Modeli fit ettik
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

#######################
# GAMMA-GAMMA Modelinin Kurulması
#######################

ggf = GammaGammaFitter(penalizer_coef=0.01) #Modeli kurduk
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])#Modeli fit ettik

#######################
# BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
#######################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left") #Final df oluşturduk.

#1-100 arası Transform
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_final[["clv"]])
cltv_final["SCALED_CLTV"] = scaler.transform(cltv_final[["clv"]]) #Scale edilmiş cltv değerini df'e ekledik.

cltv1 = ggf.customer_lifetime_value(bgf,
                                    cltv_df['frequency'],
                                    cltv_df['recency'],
                                    cltv_df['T'],
                                    cltv_df['monetary'],
                                    time=1,  # months
                                    freq="W",  # T haftalık
                                    discount_rate=0.01)

rfm_cltv1_final = cltv_df.merge(cltv1, on="Customer ID", how="left")

cltv12 = ggf.customer_lifetime_value(bgf,
                                     cltv_df['frequency'],
                                     cltv_df['recency'],
                                     cltv_df['T'],
                                     cltv_df['monetary'],
                                     time=12,  # months
                                     freq="W",  # T haftalık
                                     discount_rate=0.01)
rfm_cltv12_final = cltv_df.merge(cltv12, on="Customer ID", how="left")

#Aksiyon alabilmek için 1 aylık ve 12 aylık tahminlerde ilk 10 müşteriyi sıraladık.
rfm_cltv1_final.sort_values("clv", ascending=False).head(10)
rfm_cltv12_final.sort_values("clv", ascending=False).head(10)

#Scale edilmiş cltv skorlarına göre müşterileri segmentlere ayırdık.
cltv_final["cltv_segment"] = pd.qcut(cltv_final["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])


cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)
