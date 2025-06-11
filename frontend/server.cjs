// ws-server.js
const WebSocket = require('ws');

const wss = new WebSocket.Server({ port: 8765 });
let frontendSocket = null;

wss.on('connection', (ws, req) => {
  const ip = req.socket.remoteAddress;
  console.log(`Client connected: ${ip}`);

  if (!frontendSocket) {
    console.log('Frontend connected');
    // console.log(ws)
    frontendSocket = ws;
  } else {
    console.log('Backend connected');
    ws.on('message', (message, isBinary) => {
      console.log('Received a message');
      console.log('isBinary:', isBinary);
      if (frontendSocket && frontendSocket.readyState === WebSocket.OPEN) {
        const payload = isBinary
          ? message.toString('base64')
          : message.toString();
        frontendSocket.send(payload);
      }
    });
  }

  ws.on('close', () => {
    if (ws === frontendSocket) {
      console.log('Frontend disconnected');
      frontendSocket = null;
    }
  });
});
