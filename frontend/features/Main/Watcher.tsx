import React from "react";
import Grid from "@mui/material/Grid";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Image from "next/image";
import watcher from "../../assets/logo.png";

type Props = {};

function Watcher({}: Props) {

  return (
    <>
      <Grid
        container
        columnSpacing={"24px"}
        direction="row"
        justifyContent="center"
        alignItems="center"
        rowGap={"56px"}
        mt={"220px"}
      >
        <Grid xs={6} px={2}>
          <Box sx={{ maxWidth: "608px" }}>
            <Typography
              fontFamily={"Poppins"}
              fontStyle={"Bold"}
              fontWeight={"600"}
              fontSize={"120px"}
              lineHeight={"180px"}
              color={"#FFFFFF"}
            >
              What is <span style={{ color: "#008DFF" }}>AI DOCTOR</span>
            </Typography>
          </Box>
          <Box sx={{ marginTop: "56px" }} px={3}>
            <Typography
              fontFamily={"Poppins"}
              fontStyle={"Medium"}
              fontSize={"24px"}
              lineHeight={"36px"}
              color={"#FFFFFF"}
            >
             Welcome to AI Doctor, your one-stop solution for all your health and wellness needs. We are an innovative platform that leverages the latest advancements in Artificial Intelligence to provide personalized and accurate health advice and solutions. With our advanced technology, you can get instant access to information and support, no matter where you are. Whether you are looking for information on a specific health condition, want to track your fitness goals, or need advice on healthy lifestyle habits, AI Doctor has got you covered. Our team of experts is dedicated to delivering the best experience possible and helping you lead a healthier life. Start your journey to better health today with AI Doctor.
            </Typography>
          </Box>
        </Grid>

        <Grid xs={6}>
          <Box sx={{display:'flex',justifyContent:'center'}}>
            <Image
              src={watcher}
              alt="hero_part"
              object-fit="contain"
              width={600}
              height={450}
            />
          </Box>
        </Grid>
      </Grid>
    </>
  );
}

export default Watcher;
