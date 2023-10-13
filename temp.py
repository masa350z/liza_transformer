# %%
import requests
import numpy as np
import seaborn as sns
# %%
symbol = 'EURUSD'
m = 3
rik = 0.005
son = 0.1

base_dir = 'datas/simulation/{}'.format(symbol)

list01 = []
for i in range(10):
    list02 = []
    for j in range(20):
        rik = round(0.001*(i+1)/100, 3+2)
        son = round(0.01*(j+1)/100, 2+2)

        save_dir = base_dir + '/m{}_rik{}_son{}.npy'.format(m, rik, son)

        temp_array = np.load(save_dir)
        kane = temp_array[:, 1][-1]

        list02.append(kane)
    list01.append(list02)
# %%
list01 = np.array(list01)
# %%
sns.heatmap(list01)

# %%
params = {'Accept': '*/*',
          'Accept-Encoding': 'gzip, deflate, br',
          'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
          'Content-Encoding': 'gzip',
          'Content-Length': '546','
          'Content-Type': 'application/x-ndjson',
          'Origin': 'https://web.thinktrader.com',
          'Referer': 'https://web.thinktrader.com/',
          'Sec-Ch-Ua':'"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"'
Sec-Ch-Ua-Mobile:
?0
Sec-Ch-Ua-Platform:
"Windows"
Sec-Fetch-Dest:
empty
Sec-Fetch-Mode:
cors
Sec-Fetch-Site:
cross-site
User-Agent:
Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36
# %%
res = requests.post(
    'https://tm-monitoring.thinkmarkets.com/intake/v2/rum/events')
# %%
import requests
import json
import gzip
import io

# ヘッダー情報を設定
headers = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
    'Content-Type': 'application/x-ndjson',
    'Origin': 'https://web.thinktrader.com',
    'Referer': 'https://web.thinktrader.com/',
    'Sec-Ch-Ua': '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'cross-site',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
}
payload = b'\x8fRM\x8f\xda0\x10\xbd\xef\xaf@>\xb5\x12\x84`\x12\x96p\xaaZ\xf5P\x89\xde\xaa\xed1\x9a\xd8\x13\xf0&\xb6#\x7f\xc0V+\xfe{\xc7\th\xe9\x81KO\xb6\xdf<?\x19\xbf\xe7w\xa61\x00\x84l\xf7\xce<\xba\x93\x12\x98\xb64\xad\xec76\xb3_\x0e$:6g\'t^YC\xf0*\xe3\x9blE\x10\x1c\xd0\x84;\xbe\x8bz\xf1\xea\xff\xe1\x96\xd9\x8ag9\xbb\xccY\x0f\xe6\x10\xe9\xc6\x1d\xff\x15N\xe0\x85SCH\x044\'\xe5\xac\xd1\xa3&\x1d\x9c\x95Q\x84\xa4r\xb9\\\x9e\xdeYp`<L\x08I(I\xa4f\xcb+YVM\xb5.7E\xd96\xd4\x9ah\x02\xeb\xb1\xba)\x04<\x97m\xde\x94\xdb"\xaf\x8a\xb6\xca[.Vk\xce\xb7\x12\x1b\xc89\xb1\xaf\x83|'

# データを設定 (この例では仮のデータを使用)
data = json.dumps({"event": "example_event", "value": "example_value"})

# データをgzipで圧縮
buf = io.BytesIO()
with gzip.GzipFile(fileobj=buf, mode='wb') as f:
    f.write(data.encode('utf-8'))
gzip_data = buf.getvalue()

# Content-Encodingヘッダーを追加
headers['Content-Encoding'] = 'gzip'
headers['Content-Length'] = str(len(payload)),

# URLを設定
url = 'https://tm-monitoring.thinkmarkets.com/intake/v2/rum/events'

# POSTリクエストを送信
response = requests.post(url, headers=headers, data=payload)

# レスポンスを表示 (オプショナル)
print(response.status_code)
print(response.json())

# %%
response.text