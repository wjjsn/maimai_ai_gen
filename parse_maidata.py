import re
import requests
import json

def download_maimai_songs_data(file_path="songs.json"):
    """
    使用同步请求获取 maimai 全量数据并保存为本地 JSON 文件
    """
    url = "https://www.diving-fish.com/api/maimaidxprober/music_data"
    
    try:
        print(f"正在从 {url} 获取数据...")
        response = requests.get(url, timeout=30)
        
        # 检查 HTTP 状态码是否为 200
        response.raise_for_status()
        
        # 解析并重新保存，确保编码为 utf-8
        data = response.json()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        print(f"保存成功！共获取 {len(data)} 首歌曲，已存至: {file_path}")
        return True
        
    except Exception as e:
        print(f"下载失败，错误原因: {e}")
        return False

def load_songs_as_dict(file_path="songs.json"):
    """
    将本地 JSON 文件解析为以标题为 Key 的字典对象
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 将列表转换为字典，方便通过标题直接访问
        # key 是标题，value 是整首歌曲的详细信息对象
        songs_dict = {song['title']: song for song in data}
        
        print(f"成功解析 {len(songs_dict)} 首歌曲数据。")
        return songs_dict
    except FileNotFoundError:
        print("错误：找不到 songs.json 文件，下载中")
        download_maimai_songs_data()
        return load_songs_as_dict(file_path)
    except Exception as e:
        print(f"解析失败: {e}")
        return {}

songs = load_songs_as_dict()
def get_level(file_path:str,level:int)->float:
    pattern_title=f'&title=(.*)'
    with open(file_path, "r",encoding="utf-8") as f:
        text = f.read()
    matches = re.findall(pattern_title, text)
    title=matches[0]
    if title in songs:
        song_obj = songs[title]
        return float(song_obj['ds'][level-2])#在游戏中level1实际不存在？所以应该-2？
    else:
        print(f"未找到歌曲: {title}")
        return 0

def parse_note_time(file_path:str,level:int)->list:
    pattern = f'&inote_{level}=(.*?)E'
    pattern_level=f'&lv_{level}=(.*)'

    with open(file_path, "r",encoding="utf-8") as f:
        text = f.read()
    matches = re.findall(pattern_level, text)
    print(f'level {level}: {matches[0]}')
    matches = re.findall(pattern, text,re.DOTALL)
    # print(matches[0])
    track = matches[0]

    pattern = r'\((.*?)\)\{(.*?)\}'
    matches = re.search(pattern, track)
    BPM=float(matches.group(1))
    base_beat=int(matches.group(2))
    base_note_length=60.0/BPM*(base_beat/4.0)
    # print("BPM:",BPM,"base_beat:",base_beat,"base_note_length:",base_note_length)

    pattern = r'.*\{(.*?)\}(.*)'
    current_time = 0

    note_time:list=[]

    for count, line in enumerate(track.splitlines(), start=1):
        # print(f"第{count},{line}")
        if line=='':
            continue
        matches = re.match(pattern, line)
        beat = int(matches.group(1))
        note = matches.group(2)
        result = note.strip().split(',')
        result.pop()
        for i, item in enumerate(result):
            current_time += 240.0/(BPM*beat)
            if(item != ''):
                note_time.append(current_time)
                # print(item,current_time)
    # print(note_time)
    return note_time

if __name__ == "__main__":
    get_level(r"MIRROR of MAGIC\maidata.txt",5)
    parse_note_time(r"MIRROR of MAGIC\maidata.txt",5)