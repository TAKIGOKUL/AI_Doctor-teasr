import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import React, { useEffect, useState } from "react";
import HomeStyles from "../../styles/Home.module.css";
import { useRouter } from "next/router";
import axios from "axios";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Alert from '@mui/material/Alert';

function Form() {
  const router = useRouter();
  const [data, setData] = useState();

  const [formData, setFormData] = useState({});
  const [formResponse,setFormResponse] = useState()
  const [status,setStatus] = useState()

  const handleInputs = (e: {
    [x: string]: any;
    target: { name: any; value: any };
  }) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  useEffect(() => {
    const result = axios
      .get(`http://127.0.0.1:8000/disease/${router.query.name}`)
      .then((res) => {
        setData(res.data);
        setFormResponse(null)
      });
  }, [router.query.name]);

  const formSubmitHandler = async () => {
    const result = await axios
      .post(`http://127.0.0.1:8000/predict/${router.query.name}`, formData)
      .then((res) => {
        setFormResponse(res.data.prediction)
        setStatus(res.data.status)
        console.log(res.data.status)
      });
  };

  return (
    <div className={HomeStyles.body}>
      {formResponse == null ? <>
        <Typography
        fontStyle={"Bold"}
        fontWeight={"700"}
        fontSize={"96px"}
        lineHeight={"144px"}
        color={"#FFFFFF"}
        sx={{ paddingTop: "78px", display: "flex", justifyContent: "center" }}
      >
       Check with the {`${router.query.name}`.replaceAll("_"," ")} form
      </Typography>
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          columnGap: "138px",
          cursor: "pointer",
        }}
      >
        {data ? (
          <>
            {data.map((item: string, index: React.Key | null | undefined) => {
              return (
                <>
                  <Box
                    key={index}
                    sx={{
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                      marginTop: "25px",
                      px: 1,
                    }}
                  >
                    <Typography
                      fontFamily={"Poppins"}
                      fontStyle={"Medium"}
                      fontSize={"24px"}
                      lineHeight={"36px"}
                      color={"#FFFFFF"}
                      width={"450px"}
                    >
                      {item.replaceAll("_", " ").toUpperCase()} : 
                    </Typography>

                    <TextField
                      id="margin-none"
                      multiline
                      inputProps={{ style: { color: "#FFFFFF" ,minWidth:'500px'} }}
                      name={item}
                      value={formData[item]}
                      onChange={handleInputs}
                      sx={{
                        background: "#34353A",
                        border: "0.2px solid #34353A",
                        borderRadius: "8px",
                        marginTop: "8px",
                      }}
                      placeholder="Please enter the number between 0-10"
                    />
                  </Box>
                </>
              );
            })}
          </>
        ) : null}
      </Box>
      
      <Box sx={{display:'flex',justifyContent:'center',padding:'50px 0'}}>
      <Button
        sx={{
          border: "1px solid #008DFF",
          borderRadius: "10px",
          background: "#008DFF",
          width: "462px",
          height: "64px",
        }}
        onClick={formSubmitHandler}
      >
        <Typography
          fontFamily={"Poppins"}
          fontStyle={"SemiBold"}
          fontWeight={"600"}
          fontSize={"32px"}
          lineHeight={"48px"}
          color={"#FFFFFF"}
        >
          Upload
        </Typography>
      </Button>
      </Box>
      </>: <div style={{minHeight:'100vh',display:'flex',justifyContent:'center',alignItems:'center'}}>
      <Alert severity={status}>{formResponse}</Alert>
      </div>}
      
      
    </div>
  );
}

export default Form;
