"use client";

import React, { useEffect, useRef, useState } from "react";

const Camera = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const displayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [socket, setSocket] = useState<WebSocket | null>(null);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    setSocket(ws);

    // Get video stream
    const getVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing the camera: ", err);
      }
    };

    getVideo();

    // Clean up
    return () => {
      const video = videoRef.current;
      if (video && video.srcObject) {
        const stream = video.srcObject as MediaStream;
        const tracks = stream.getTracks();
        tracks.forEach((track) => track.stop());
      }
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
      requestAnimationFrame(sendFrame);
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
            // Clear the canvas before drawing the new image
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
      <h2>Processed Camera Feed</h2>
      <video ref={videoRef} autoPlay playsInline style={{ display: "none" }} />
      <div className="flex flex-row ">
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
