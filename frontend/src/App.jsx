import { useEffect, useRef } from 'react';

function VideoStream() {
  const imgRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8765');

    ws.onopen = () => console.log("WebSocket connection opened");
    ws.onerror = (err) => console.error("WebSocket error", err);
    ws.onclose = () => console.warn("WebSocket closed");
    
    ws.onmessage = (event) => {
      const img = `data:image/jpeg;base64,${event.data}`;
      if (imgRef.current) {
        imgRef.current.src = img;
      }
    };
    return () => ws.close();
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-screen p-4">
      <h1 className="text-xl font-bold mb-4">Dora Video Stream</h1>
      <img ref={imgRef} className="rounded shadow-lg max-w-full" alt="Video stream" />
    </div>
  );
}

export default VideoStream;