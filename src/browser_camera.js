// Keep the media stream alive across Streamlit reruns. Unmounting releases it.
class DetectorCamera extends HTMLElement {
  connectedCallback() {
    this.status = document.createElement("span");
    this.status.setAttribute("role", "status");
    this.status.textContent = "Permite el acceso a la cámara en tu navegador.";
    this.replaceChildren(this.status);
    this.video = document.createElement("video");
    this.video.muted = true;
    this.video.playsInline = true;
    this.canvas = document.createElement("canvas");
    this.open();
  }

  async open() {
    const video = this.video;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false
      });
      if (!this.isConnected || video !== this.video) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      video.srcObject = stream;
      await video.play();
      this.status.textContent = "Cámara conectada.";
      this.schedule();
    } catch (error) {
      this.status.textContent = error.name === "NotAllowedError"
        ? "Permite la cámara en los permisos del sitio y vuelve a activar Analizar."
        : "No se pudo iniciar la cámara. Comprueba que esté disponible y vuelve a activar Analizar.";
    }
  }

  schedule() {
    clearTimeout(this.timer);
    if (!this.isConnected || !this.video.videoWidth || this.pending) return;
    this.timer = setTimeout(() => {
      const scale = Math.min(1, 640 / Math.max(this.video.videoWidth, this.video.videoHeight));
      this.canvas.width = Math.max(1, Math.round(this.video.videoWidth * scale));
      this.canvas.height = Math.max(1, Math.round(this.video.videoHeight * scale));
      this.canvas.getContext("2d").drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
      this.pending = crypto.randomUUID();
      this.send("frame", { id: this.pending, jpeg: this.canvas.toDataURL("image/jpeg", 0.75).split(",")[1] });
    }, 250);
  }

  disconnectedCallback() {
    clearTimeout(this.timer);
    this.video.srcObject?.getTracks().forEach(track => track.stop());
    this.video.srcObject = null;
    this.pending = null;
  }
}

export default function({ parentElement, data, setTriggerValue }) {
  if (!customElements.get("vigia-browser-camera")) {
    customElements.define("vigia-browser-camera", DetectorCamera);
  }
  const camera = parentElement.querySelector("vigia-browser-camera");
  camera.send = setTriggerValue;
  if (camera.pending && data.ack === camera.pending) {
    camera.pending = null;
    camera.schedule();
  }
}
