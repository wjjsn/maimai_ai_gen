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

def format_maidata(text:str):
    # 1. 预处理：删除 &inote_x= 后面的换行，确保前缀与内容接通
    text = re.sub(r'(&inote_\d+=)\n+', r'\1', text)
    
    lines = text.splitlines()
    temp_result = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 规则 A：'E' 和 '&' 开头的行先入列
        if line == 'E' or line.startswith('&'):
            temp_result.append(line)
            continue
            
        # 规则 B：处理包含大括号的行
        if '{' in line:
            # 拆分大括号前的内容（如 (200)）和大括号后的内容
            parts = re.split(r'(?=\{)', line, maxsplit=1)
            prefix, bracket_part = parts[0], parts[1]
            
            if prefix:
                # 如果有前缀，必须合并到上一行
                if temp_result:
                    temp_result[-1] += prefix
                else:
                    temp_result.append(prefix)
            
            # 关键判断：如果上一行是 & 开头的行，且还没有大括号，则合并
            if temp_result and temp_result[-1].startswith('&') and '{' not in temp_result[-1]:
                temp_result[-1] += bracket_part
            else:
                # 否则，大括号必须换行（保证一行最多一个大括号）
                temp_result.append(bracket_part)
        else:
            # 规则 C：纯杂质（逗号、括号等），合并到上一行
            if temp_result:
                temp_result[-1] += line
            else:
                temp_result.append(line)

    return "\n".join(temp_result)

def parse_note_time(file_path:str,level:int)->list:
    pattern = f'&inote_{level}=(.*?)E'
    pattern_level=f'&lv_{level}=(.*)'

    with open(file_path, "r",encoding="utf-8") as f:
        text = f.read()
    text=format_maidata(text)
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