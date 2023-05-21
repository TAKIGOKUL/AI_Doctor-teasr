import * as React from "react";
import Button from "@mui/material/Button";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle";
import Paper, { PaperProps } from "@mui/material/Paper";
import Draggable from "react-draggable";
import TextField from "@mui/material/TextField";
import axios from "axios";
import Typography from "@mui/material/Typography";
import Image from "next/image";

import chatbot_logo from "../../assets/chatbot.png"

interface IAllChats {
  question: string;
  response: string;
}

function PaperComponent(props: PaperProps) {
  return (
    <Draggable
      handle="#draggable-dialog-title"
      cancel={'[class*="MuiDialogContent-root"]'}
    >
      <Paper {...props} />
    </Draggable>
  );
}

export default function Bot() {

  const [open, setOpen] = React.useState(false);
  const [question, setQuestion] = React.useState("");

  const [allChats, setAllChats] = React.useState<IAllChats[]>([]);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const callingBotApi = () => {
    axios
      .get(`http://127.0.0.1:8000/bot?text=${question}`)
      .then((res) => {
        const data = res.data.medboat
        setAllChats([...allChats, { question, response: data }]);
      });
  };

  const handleInputs = (e: { target: { value: React.SetStateAction<string>; }; }) => {
    setQuestion(e.target.value)
  }
  
  console.log(allChats)

  return (
    <>
      <div onClick={handleClickOpen} style={{cursor:"pointer"}}>
      <Image src={chatbot_logo} alt="logo" />
      </div>

      <Dialog
        open={open}
        onClose={handleClose}
        PaperComponent={PaperComponent}
        aria-labelledby="draggable-dialog-title"
      >
        <DialogTitle style={{ cursor: "move" }} id="draggable-dialog-title">
          Get the Answer of all your doubts with our bot.
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            {allChats.map((item,index) => {
              return (
                <div style={{marginTop:'20px'}}>
                <Typography
                      fontFamily={"Poppins"}
                      fontStyle={"Medium"}
                      fontSize={"18px"}
                      lineHeight={"24px"}
                    >
                      User : {item.question}
                    </Typography>

                    <Typography
                      fontFamily={"Poppins"}
                      fontStyle={"Medium"}
                      fontSize={"18px"}
                      lineHeight={"24px"}
                    >
                      Bot : {item.response}
                    </Typography>
                </div>
              )
            })}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <TextField
            id="margin-none"
            multiline
            inputProps={{ style: { minWidth: "500px" } }}
            name={"message"}
            value={question}
            onChange={handleInputs}
            placeholder="Please enter your question to ask"
          />
          <Button autoFocus onClick={handleClose}>
            Cancel
          </Button>
          <Button onClick={callingBotApi}>Post</Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
