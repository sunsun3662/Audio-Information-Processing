#coding=utf-8

import os
import torch
import librosa
import numpy as np
import matplotlib.cm as cm
from PIL import Image, ImageDraw
import soundfile as sf
from ns3_codec import FACodecEncoder, FACodecDecoder


#### 实验：使用Facodec实现语音离散化、特征解耦与重构
def main():

    #### Step1: 实例化FACodecEncoder和FACodecDecoder并加载预训练模型
        ### 此处实例化fa_encoder和fa_decoder并加载预训练好的模型权重
    
    fa_encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    # 统一使用脚本所在目录，避免在不同工作目录下找不到文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, "ckpt")
    samples_dir = os.path.join(base_dir, "samples")

    fa_encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, "ns3_facodec_encoder.bin")))
    fa_decoder.load_state_dict(torch.load(os.path.join(ckpt_dir, "ns3_facodec_decoder.bin")))

    
    #### Step2: 加载测试音频文件
    def load_audio(wav_path, sr=16000):
        wav = librosa.load(wav_path, sr=sr)[0]
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
        return wav, wav_t

    def plot_waveform(wav, sr, title, out_path):
        # 使用 PIL 生成波形图，避免 matplotlib 在 Windows 上崩溃
        width, height = 1000, 300
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), title, fill="black")

        wav = wav / (np.max(np.abs(wav)) + 1e-9)
        mid = height // 2
        scale = height * 0.45
        samples_per_px = max(1, len(wav) // width)

        for x in range(width):
            start = x * samples_per_px
            end = min(len(wav), (x + 1) * samples_per_px)
            seg = wav[start:end]
            if seg.size == 0:
                continue
            y_min = int(mid - np.max(seg) * scale)
            y_max = int(mid - np.min(seg) * scale)
            draw.line((x, y_min, x, y_max), fill="black")

        img.save(out_path)

    def plot_spectrogram(wav, sr, title, out_path):
        # 彩色频谱图：用 colormap 上色，再用 PIL 保存
        D = librosa.stft(wav)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
        cmap = cm.get_cmap("magma")
        rgba = cmap(S_norm)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        rgb = np.flipud(rgb)
        img = Image.fromarray(rgb, mode="RGB").resize((1000, 300), Image.BILINEAR)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), title, fill="white")
        img.save(out_path)
    
    # 固定采样率为 16kHz
    sr = 16000
    test_wav_file = "speaker2"
    test_wav_np, test_wav = load_audio(os.path.join(samples_dir, f"{test_wav_file}.wav"), sr=sr)
    print("Test Audio Shape: ", test_wav.shape)
    

    #### Step3: 使用Facodec对音频进行离散化和解耦处理
    fa_encoder.eval()
    fa_decoder.eval()
    with torch.no_grad():

        #### Step3.1: 使用fa_encoder对音频进行嵌入编码
        encoder_out = fa_encoder(test_wav)
        print("Encoder_out Shape: ", encoder_out.shape)

        #### Step3.2: 观察fa_decoder对音频进行离散化的效果
        _, vq_id, _, _, spk_embs = fa_decoder(encoder_out, eval_vq=False, vq=True)
        prosody_code = vq_id[:1]
        print("Prosody Code Shape:", prosody_code.shape)
        cotent_code = vq_id[1:3]
        print("Cotent Code Shape:", cotent_code.shape)
        detail_code = vq_id[3:]
        print("Residual Code Shape:", detail_code.shape) ### tips: 此处batch_size可能不等于1，注意第二维度
        
        #### Step3.3: 对比：将离散化后的向量通过fa_decoder对音频进行解码重构
        _, _, _, _, spk_embs = fa_decoder(encoder_out, eval_vq=False, vq=True)
        _, _, _, quantized = fa_decoder.quantize(encoder_out)
        prosody = quantized[0]
        print("Prosody Embedding Shape:", prosody.shape)
        content = quantized[1]
        print("Content Embedding Shape:", content.shape)
        detail = quantized[2]
        print("Detail Embedding Shape:", detail.shape)
        spk_embs = spk_embs
        print("Speaker Embedding Shape:", spk_embs.shape)
        
        #### Step3.4: 混合四个特征向量, 使用fa_decoder进行重构音频
        all_embs = prosody + content + detail
        rec_wav = fa_decoder.inference(all_embs, spk_embs)


    #### Step4: 保存重构后的音频
    print("Reconstruct Audio Shape: ", rec_wav.shape)
    rec_np = rec_wav[0][0].cpu().numpy()
    sf.write(os.path.join(samples_dir, f"{test_wav_file}_rec.wav"), rec_np, sr)
    try:
        # 保存原始/重构的波形图和频谱图
        plot_waveform(test_wav_np, sr, "Original Waveform", os.path.join(samples_dir, f"{test_wav_file}_waveform.png"))
        plot_waveform(rec_np, sr, "Reconstructed Waveform", os.path.join(samples_dir, f"{test_wav_file}_rec_waveform.png"))
        plot_spectrogram(test_wav_np, sr, "Original Spectrogram", os.path.join(samples_dir, f"{test_wav_file}_spec.png"))
        plot_spectrogram(rec_np, sr, "Reconstructed Spectrogram", os.path.join(samples_dir, f"{test_wav_file}_rec_spec.png"))
    except Exception as exc:
        print("Plotting failed:", type(exc).__name__, exc)
    print("Successfully Reconstruct!")

    #### Step5: voice conversion (optional when user.wav exists)
    source_wav_file = "speaker1"
    user_wav_file = "user"
    user_wav_path = os.path.join(samples_dir, f"{user_wav_file}.wav")
    if os.path.exists(user_wav_path):
        # 当 samples/user.wav 存在时，执行音色转换
        source_np, source_wav = load_audio(os.path.join(samples_dir, f"{source_wav_file}.wav"), sr=sr)
        user_np, user_wav = load_audio(user_wav_path, sr=sr)
        with torch.no_grad():
            encoder_out_source = fa_encoder(source_wav)
            encoder_out_user = fa_encoder(user_wav)
            _, _, _, _, spk_embs_user = fa_decoder(encoder_out_user, eval_vq=False, vq=True)
            _, _, _, quantized_source = fa_decoder.quantize(encoder_out_source)
            prosody_source = quantized_source[0]
            content_source = quantized_source[1]
            detail_source = quantized_source[2]
            all_embs_source = prosody_source + content_source + detail_source
            rec_wav_with_user = fa_decoder.inference(all_embs_source, spk_embs_user)

        rec_user_np = rec_wav_with_user[0][0].cpu().numpy()
        sf.write(
            os.path.join(samples_dir, f"{source_wav_file}_rec_with_{user_wav_file}.wav"),
            rec_user_np,
            sr,
        )
        try:
            plot_waveform(source_np, sr, "Source Waveform", os.path.join(samples_dir, f"{source_wav_file}_waveform.png"))
            plot_waveform(
                rec_user_np,
                sr,
                "Converted Waveform",
                os.path.join(samples_dir, f"{source_wav_file}_to_{user_wav_file}_waveform.png"),
            )
            plot_spectrogram(source_np, sr, "Source Spectrogram", os.path.join(samples_dir, f"{source_wav_file}_spec.png"))
            plot_spectrogram(
                rec_user_np,
                sr,
                "Converted Spectrogram",
                os.path.join(samples_dir, f"{source_wav_file}_to_{user_wav_file}_spec.png"),
            )
        except Exception as exc:
            print("Plotting failed:", type(exc).__name__, exc)
        print("Successfully Convert Voice!")
    else:
        print("No user wav found, skip voice conversion: ", user_wav_path)

if __name__ == "__main__":
    main()