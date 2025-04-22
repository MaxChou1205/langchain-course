from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

if __name__ == "__main__":
    print("first llm")

    information = """
     方濟各（拉丁語：Franciscus；義大利語：Francesco；西班牙語：Francisco；1936年12月17日—2025年4月21日），本名喬治·馬里奧·伯格里奧（西班牙語：Jorge Mario Bergoglio），是天主教會第266任教宗，耶穌會會士，義大利裔阿根廷人，能說流利的拉丁語、西班牙語、義大利語和德語[1]。1958年加入耶穌會，1969年晉鐸，1997年晉牧任布宜諾斯艾利斯總主教，並在2001年由時任教宗若望保祿二世冊封為樞機。他在2013年3月13日獲選為教宗，成為首位出身於拉丁美洲、南半球與耶穌會的教宗，也是繼額我略三世後1282年以來首位非歐洲出身的教宗。他也是第一位沒有居住在梵蒂岡教宗官邸的教宗，而是選擇梵蒂岡的宿舍設施聖瑪爾大之家的客房作為住處。2025年4月21日，於聖瑪爾大之家因中風導致心臟衰竭而逝世。
    """

    summary_template = """
    given the following information {information} about a person i want to create:
    1. A short summary
    2. two interesting facts about the person
    3. response in Chinese
    """

    summary_prompt = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(model="gemma3:latest")

    chain = summary_prompt | llm | StrOutputParser
    res = chain.invoke(input={"information": information})

    print(res)

