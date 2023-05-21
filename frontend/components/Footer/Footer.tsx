import React from "react";
import Box from "@mui/material/Box";
import Image from "next/image";
import logo from "../../assets/logo.png";
import Typography from "@mui/material/Typography";
import Toolbar from "@mui/material/Toolbar";
import Bot from "../Bot/Bot";

function Footer() {
  return (
    <>
      <Box sx={{ marginTop: "56px", background: "#000C15", padding: "101px" }}>
        <Toolbar>
          <div style={{ flexGrow: 1, display: "flex", alignItems: "center" }}>
            <Image src={logo} alt="logo" height={150} width={200}/>
            <Typography
              fontFamily={"Poppins"}
              fontStyle={"Bold"}
              fontWeight={"400"}
              fontSize={"40px"}
              lineHeight={"60px"}
              color={"#34353A"}
            >
              AI DOCTOR
            </Typography>
          </div>
          <div style={{display:'flex',justifyContent:'center',alignItems:'center',flexDirection:'column'}}>
            <Bot />

            <Typography
              fontFamily={"Poppins"}
              fontStyle={"Medium"}
              fontWeight={"400"}
              fontSize={"24px"}
              lineHeight={"36px"}
              color={"#34353A"}
              mt={3}
            >
              Copyright Ⓒ 2022 Watcher All rights reserved
            </Typography>
          </div>
        </Toolbar>
      </Box>
    </>
  );
}

export default Footer;
