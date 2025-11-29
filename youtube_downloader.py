import yt_dlp
import os
import uuid
import traceback 

def download_youtube_video(video_url, output_folder='uploads', start_time=None, end_time=None):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    unique_name = str(uuid.uuid4())

    try:
        temp_opts = {'quiet': True, 'noplaylist': True}
        with yt_dlp.YoutubeDL(temp_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            ext = info.get('ext', 'mp4') 
    except Exception as e:
        print(f"'{video_url}'의 정보 가져오기 실패: {e}")
        ext = 'mp4' 

    if start_time and end_time:
        ext = 'mp4'
        
    final_path = os.path.join(output_folder, f"{unique_name}.{ext}")

    
    if start_time and end_time:
        print(f"'{video_url}'의 {start_time}~{end_time} 구간 다운로드를 시작합니다.")
        ydl_opts = {
            'format': 'mp4/best', 
            'outtmpl': final_path, 
            'quiet': True,
            'noplaylist': True,
            'download_sections': f"*time={start_time}-{end_time}",
            'force_keyframes_at_cuts': True,
            'postprocessor_args': [ 
                '-ss', start_time,
                '-to', end_time,
                '-c', 'copy' 
            ]
        }
    else:
        print(f"'{video_url}' 전체 다운로드를 시작합니다")
        ydl_opts = {
            'format': 'best[ext=mp4]/best', 
            'outtmpl': final_path, 
            'keepvideo': False,
            'quiet': True, 
            'noplaylist': True,
        }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if os.path.exists(final_path):
            print(f"다운로드 성공: {final_path}")
            return final_path
        else:
            print(f"경고: 예상 경로({final_path})에 파일이 없습니다.")
            for f in os.listdir(output_folder):
                if f.startswith(unique_name):
                    actual_path = os.path.join(output_folder, f)
                    print(f"실제 생성된 파일 경로로 반환: {actual_path}")
                    return actual_path
            
            print(f"오류: {unique_name}으로 시작하는 다운로드된 파일을 찾을 수 없습니다.")
            return None

    except Exception as e:
        print(f"'{video_url}' 다운로드 중 예외 발생:")
        traceback.print_exc()
        return None