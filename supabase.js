import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = "https://mmumwcgdurxiimymthsl.supabase.co";
const SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1tdW13Y2dkdXJ4aWlteW10aHNsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzg5MzAyNTgsImV4cCI6MjA1NDUwNjI1OH0.dXR2-Xj2clV5lKCiDSZBVMPFJPBKH6dRFA1G5ubk0d0";

export const supabase = createClient(SUPABASE_URL, SUPABASE_KEY);

export const saveMessage = async (userMessage, botResponse) => {
  await supabase.from("messages").insert([
    { user_message: userMessage, bot_response: botResponse },
  ]);
};
