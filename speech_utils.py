import speech_recognition as sr
import streamlit as st

def record_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 マイクから話してください...")
        try:
            audio = r.listen(source, timeout=5)
            text = r.recognize_google(audio, language="ja-JP")
            st.success(f"✅ 音声認識結果：{text}")
            return text
        except sr.UnknownValueError:
            st.error("❌ 音声を認識できませんでした。")
        except sr.RequestError as e:
            st.error(f"❌ 認識エラー: {e}")
        except sr.WaitTimeoutError:
            st.warning("⏰ 音声が入力されませんでした。")
    return None
