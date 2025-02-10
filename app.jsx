import { useState } from "react";
import ChatButton from "./components/ChatButton";
import ChatWindow from "./components/ChatWindow";
import { sendMessageToLLM } from "./services/api";

function App() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);

  const handleSend = async (message) => {
    const response = await sendMessageToLLM(message);
    setMessages([...messages, { user: message, bot: response }]);
  };

  return (
    <>
      {open && <ChatWindow onClose={() => setOpen(false)} onSend={handleSend} />}
      <ChatButton onClick={() => setOpen(true)} />
    </>
  );
}

export default App;
