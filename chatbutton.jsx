import { useState } from "react";
import { FaRobot } from "react-icons/fa";

const ChatButton = ({ onClick }) => {
  return (
    <button
      onClick={onClick}
      className="fixed bottom-6 right-6 bg-blue-500 text-white p-4 rounded-full shadow-lg hover:bg-blue-700 transition"
    >
      <FaRobot size={24} />
    </button>
  );
};

export default ChatButton;
