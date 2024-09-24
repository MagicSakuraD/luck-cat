"use client";

import React, { useEffect, useRef, useState } from "react";

const Camera = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  // Get video from file input
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && videoRef.current) {
      const url = URL.createObjectURL(file);
      videoRef.current.src = url;
      videoRef.current.play();
    }
  };

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    setSocket(ws);

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  useEffect(() => {
    if (!socket) return;

    const sendFrame = () => {
      const video = videoRef.current;
      const canvas = captureCanvasRef.current;
      if (video && canvas) {
        const context = canvas.getContext("2d");
        if (context) {
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          const imageData = canvas.toDataURL("image/jpeg");
          socket.send(imageData);
        }
      }
      setTimeout(sendFrame, 100); // 每100ms发送一帧
    };

    socket.onopen = () => {
      console.log("WebSocket connected");
      sendFrame();
    };

    socket.onmessage = (event) => {
      const img = new Image();
      img.onload = () => {
        const canvas = displayCanvasRef.current;
        if (canvas) {
          const context = canvas.getContext("2d");
          if (context) {
            context.clearRect(0, 0, canvas.width, canvas.height);
            context.drawImage(img, 0, 0, canvas.width, canvas.height);
          }
        }
      };
      img.src = event.data;
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    socket.onclose = () => {
      console.log("WebSocket disconnected");
    };
  }, [socket]);

  return (
    <div>
      <h2>Processed Video Feed</h2>
      <input type="file" accept="video/*" onChange={handleFileChange} />
      <video ref={videoRef} autoPlay playsInline style={{ display: "none" }} />
      <div className="flex flex-row">
        <canvas ref={captureCanvasRef} width={640} height={480} />
        <canvas
          ref={displayCanvasRef}
          width={640}
          height={480}
          style={{ width: "100%", maxWidth: "640px" }}
        />
      </div>
    </div>
  );
};

export default Camera;
