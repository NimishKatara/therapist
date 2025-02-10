import { useState } from "react";

const ChatWindow = ({ onClose, onSend }) => {
  const [message, setMessage] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim()) {
      onSend(message);
      setMessage("");
    }
  };

  return (
    <div className="fixed bottom-16 right-6 w-80 bg-white shadow-lg p-4 rounded-lg">
      <div className="text-right">
        <button onClick={onClose} className="text-red-500 text-xl">&times;</button>
      </div>
      <div className="h-40 overflow-y-auto border p-2 mb-2"> {/* Chat Messages Here */}</div>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type a message..."
          className="w-full p-2 border rounded"
        />
        <button type="submit" className="w-full bg-blue-500 text-white p-2 mt-2 rounded">
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatWindow;
